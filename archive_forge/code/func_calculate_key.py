import operator
import sys
import traceback
import weakref
@classmethod
def calculate_key(cls, target):
    """Calculate the reference key for this reference.

        Currently this is a two-tuple of the id()'s of the target
        object and the target function respectively.
        """
    return (id(get_self(target)), id(get_func(target)))