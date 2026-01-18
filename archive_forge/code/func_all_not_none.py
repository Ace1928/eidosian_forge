import jmespath
from botocore import xform_name
from .params import get_data_member
def all_not_none(iterable):
    """
    Return True if all elements of the iterable are not None (or if the
    iterable is empty). This is like the built-in ``all``, except checks
    against None, so 0 and False are allowable values.
    """
    for element in iterable:
        if element is None:
            return False
    return True