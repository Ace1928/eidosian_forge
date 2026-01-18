import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from onnx import MapProto, OptionalProto, SequenceProto, TensorProto, helper, subbyte
from onnx.external_data_helper import load_external_data_for_tensor, uses_external_data
def combine_pairs_to_complex(fa: Sequence[int]) -> List[complex]:
    return [complex(fa[i * 2], fa[i * 2 + 1]) for i in range(len(fa) // 2)]