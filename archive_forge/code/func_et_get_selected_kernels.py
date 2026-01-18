from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import yaml
from torchgen.model import NativeFunction
from torchgen.selective_build.operator import (
def et_get_selected_kernels(self, op_name: str, kernel_key: List[str]) -> List[str]:
    """
        Return a list of kernel keys that cover the used ops
        """
    if op_name not in self.et_kernel_metadata:
        return kernel_key if self.include_all_operators else []
    result_set = set()
    for model_kernel_keys in self.et_kernel_metadata[op_name]:
        key_found = False
        for key in kernel_key:
            if key != 'default' and key.split('/')[1] == model_kernel_keys.split('/')[1]:
                result_set.add(key)
                key_found = True
                break
        if not key_found:
            if 'default' not in kernel_key:
                raise Exception('Missing kernel for the model')
            else:
                result_set.add('default')
    return list(result_set)