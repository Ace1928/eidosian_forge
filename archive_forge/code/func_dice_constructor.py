import re
import pytest  # NOQA
from .roundtrip import dedent
def dice_constructor(loader, node):
    value = loader.construct_scalar(node)
    a, b = map(int, value.split('d'))
    return Dice(a, b)