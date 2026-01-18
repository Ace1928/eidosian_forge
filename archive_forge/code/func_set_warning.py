from .. import utils
from .._lazyload import rpy2
from . import conversion
import functools
@staticmethod
def set_warning():
    _ConsoleWarning.set(_ConsoleWarning.warning)