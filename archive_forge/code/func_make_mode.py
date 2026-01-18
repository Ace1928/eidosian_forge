from __future__ import print_function, unicode_literals
import typing
from typing import Iterable
import six
from ._typing import Text
def make_mode(init):
    """Make a mode integer from an initial value."""
    return Permissions.get_mode(init)