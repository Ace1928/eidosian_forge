from .. import errors
from ..decorators import only_raises
def disable_unlock(self):
    """Make an unlock call fail"""
    self.__dict__['_allow_unlock'] = False