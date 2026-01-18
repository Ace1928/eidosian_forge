import inspect
import sys
def method_name(method):
    """ Return a method's name.

    This function returns the name the method is accessed by from
    outside the class (i.e. it prefixes "private" methods appropriately).
    """
    mname = name(method)
    if is_class_private_name(mname):
        mname = '_%s%s' % (name(method.__self__.__class__), mname)
    return mname