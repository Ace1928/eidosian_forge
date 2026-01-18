from ..helpers import map_values
from ._higherorder import (
from ._impl import Mismatch
@classmethod
def fromExample(cls, example, *attributes):
    from ._basic import Equals
    kwargs = {}
    for attr in attributes:
        kwargs[attr] = Equals(getattr(example, attr))
    return cls(**kwargs)