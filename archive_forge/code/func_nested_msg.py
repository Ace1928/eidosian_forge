import traceback
from collections import namedtuple, defaultdict
import itertools
import logging
import textwrap
from shutil import get_terminal_size
from .abstract import Callable, DTypeSpec, Dummy, Literal, Type, weakref
from .common import Opaque
from .misc import unliteral
from numba.core import errors, utils, types, config
from numba.core.typeconv import Conversion
def nested_msg(literalness, e):
    estr = str(e)
    estr = estr if estr else str(repr(e)) + add_bt(e)
    new_e = errors.TypingError(textwrap.dedent(estr))
    return tmplt.format(literalness, str(new_e))