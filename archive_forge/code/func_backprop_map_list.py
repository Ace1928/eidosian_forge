from typing import Callable, List, Optional, Tuple, TypeVar
from ..model import Model
def backprop_map_list(dYs: List[OutT]) -> List[InT]:
    return [callback(dY) for callback, dY in zip(callbacks, dYs)]