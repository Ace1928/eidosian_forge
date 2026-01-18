import pytest
from traitlets import HasTraits, TraitError
from ..traittypes import SciType
def maxlen(trait, value):
    if len(value) > 10:
        raise ValueError('Too long sequence!')
    return value