from ..helpers import map_values
from ._higherorder import (
from ._impl import Mismatch
@classmethod
def byMatcher(cls, matcher, **kwargs):
    """Matches an object where the attributes match the keyword values.

        Similar to the constructor, except that the provided matcher is used
        to match all of the values.
        """
    return cls(**map_values(matcher, kwargs))