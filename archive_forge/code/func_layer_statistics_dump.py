import collections
import csv
import re
from typing import (Any, Callable, Dict, IO, Iterable, List, Mapping, Optional,
import numpy as np
from tensorflow.lite.python import convert
from tensorflow.lite.python import interpreter as _interpreter
from tensorflow.lite.python.metrics import metrics as metrics_stub  # type: ignore
from tensorflow.python.util import tf_export
def layer_statistics_dump(self, file: IO[str]) -> None:
    """Dumps layer statistics into file, in csv format.

    Args:
      file: file, or file-like object to write.
    """
    fields = ['op_name', 'tensor_idx'] + list(self._layer_debug_metrics.keys())
    if self._debug_options.layer_direct_compare_metrics is not None:
        fields += list(self._debug_options.layer_direct_compare_metrics.keys())
    fields += ['scale', 'zero_point', 'tensor_name']
    writer = csv.DictWriter(file, fields)
    writer.writeheader()
    if self.layer_statistics:
        for name, metrics in self.layer_statistics.items():
            data = metrics.copy()
            data['tensor_name'], _ = self._get_operand_name_and_index(name)
            data['tensor_idx'] = self._numeric_verify_op_details[name]['inputs'][0]
            data['op_name'] = self._quant_interpreter._get_op_details(self._defining_op[data['tensor_idx']])['op_name']
            details = self._quant_interpreter._get_tensor_details(data['tensor_idx'], subgraph_index=0)
            data['scale'], data['zero_point'] = (details['quantization_parameters']['scales'][0], details['quantization_parameters']['zero_points'][0])
            writer.writerow(data)