from .. import utils
from .._lazyload import rpy2
from . import conversion
import functools
@staticmethod
def set_builtin():
    _ConsoleWarning.set(_ConsoleWarning.builtin_warning)