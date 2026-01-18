import weakref
import importlib
from numba import _dynfunc
def lookup_environment(env_name):
    """Returns the Environment object for the given name;
    or None if not found
    """
    return Environment._memo.get(env_name)