import pickle
from typing import Any
import torch
def pyobject_to_tensor(obj: Any, fixed_buffer_size: int=0) -> torch.Tensor:
    pickled = pickle.dumps(obj)
    result: torch.Tensor = torch.ByteTensor(bytearray(pickled))
    if fixed_buffer_size:
        delta = fixed_buffer_size - len(result)
        if delta < 0:
            raise ValueError(f'message too big to send, increase `fixed_buffer_size`? - {len(result)} > {fixed_buffer_size}')
        elif delta > 0:
            result = torch.cat((result, torch.zeros(delta, dtype=torch.uint8)))
    return result