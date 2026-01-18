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
def template_info(tp):
    src_info = tp.get_template_info()
    unknown = 'unknown'
    source_name = src_info.get('name', unknown)
    source_file = src_info.get('filename', unknown)
    source_lines = src_info.get('lines', unknown)
    source_kind = src_info.get('kind', 'Unknown template')
    return (source_name, source_file, source_lines, source_kind)