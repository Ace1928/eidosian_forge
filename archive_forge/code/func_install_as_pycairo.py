import sys
from ctypes.util import find_library
from . import constants
from .ffi import ffi
from .surfaces import (  # noqa isort:skip
from .patterns import (  # noqa isort:skip
from .fonts import (  # noqa isort:skip
from .context import Context  # noqa isort:skip
from .matrix import Matrix  # noqa isort:skip
from .constants import *  # noqa isort:skip
def install_as_pycairo():
    """Install cairocffi so that ``import cairo`` imports it.

    cairoffi’s API is compatible with pycairo as much as possible.

    """
    sys.modules['cairo'] = sys.modules[__name__]