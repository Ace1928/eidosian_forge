from warnings import warn
from ..auto import tqdm as tqdm_auto
from ..std import TqdmDeprecationWarning, tqdm
from ..utils import ObjectWrapper
def builtin_iterable(func):
    """Returns `func`"""
    warn('This function has no effect, and will be removed in tqdm==5.0.0', TqdmDeprecationWarning, stacklevel=2)
    return func