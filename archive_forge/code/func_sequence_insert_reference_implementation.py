from typing import Any, List, Optional, Union
import numpy as np
from onnx.reference.op_run import OpRun
def sequence_insert_reference_implementation(sequence: Union[List[Any], np.ndarray], tensor: np.ndarray, position: Optional[np.ndarray]=None) -> List[Any]:
    seq: List[Any] = []
    if sequence is not None and (not isinstance(sequence, np.ndarray) or len(sequence.shape) > 0):
        try:
            seq.extend(sequence)
        except TypeError as e:
            raise TypeError(f'Unable to iterate on type {type(sequence)}: {sequence}.') from e
    if position is not None:
        insert_position = (position[0] + len(seq)) % len(seq)
        seq.insert(insert_position, tensor)
    else:
        seq.append(tensor)
    return seq