from collections.abc import Iterable
from collections.abc import Sequence as pySequence
from types import MappingProxyType
from .abstract import (
from .common import (
from .misc import Undefined, unliteral, Optional, NoneType
from ..typeconv import Conversion
from ..errors import TypingError
from .. import utils
def is_homogeneous(*tys):
    """Are the types homogeneous?
    """
    if tys:
        first, tys = (tys[0], tys[1:])
        return not any((t != first for t in tys))
    else:
        return False