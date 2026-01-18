from typing import Callable, List, Tuple, TypeVar, cast
from ..config import registry
from ..model import Model
from ..types import ArrayXd, ListXd, Ragged
@registry.layers('list2ragged.v1')
def list2ragged() -> Model[InT, OutT]:
    """Transform sequences to ragged arrays if necessary and return the ragged
    array. If sequences are already ragged, do nothing. A ragged array is a
    tuple (data, lengths), where data is the concatenated data.
    """
    return Model('list2ragged', forward)