from collections.abc import Iterable
from collections.abc import Sequence as pySequence
from types import MappingProxyType
from .abstract import (
from .common import (
from .misc import Undefined, unliteral, Optional, NoneType
from ..typeconv import Conversion
from ..errors import TypingError
from .. import utils
def check_field_pair(fieldpair):
    name, typ = fieldpair
    if not isinstance(name, str):
        msg = 'expecting a str for field name'
        raise ValueError(msg)
    if not isinstance(typ, Type):
        msg = 'expecting a Numba Type for field type'
        raise ValueError(msg)
    return (name, typ)