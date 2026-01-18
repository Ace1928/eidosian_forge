from .. import errors
from ..decorators import only_raises
def disable_lock_read(self):
    """Make a lock_read call fail"""
    self.__dict__['_allow_read'] = False