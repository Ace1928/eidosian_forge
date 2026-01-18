from suds import *
from logging import getLogger
def footprint(sobject):
    """
    Get the I{virtual footprint} of the object.

    This is really a count of all the significant value attributes in the
    branch.

    @param sobject: A suds object.
    @type sobject: L{Object}
    @return: The branch footprint.
    @rtype: int

    """
    n = 0
    for a in sobject.__keylist__:
        v = getattr(sobject, a)
        if v is None:
            continue
        if isinstance(v, Object):
            n += footprint(v)
            continue
        if hasattr(v, '__len__'):
            if len(v):
                n += 1
            continue
        n += 1
    return n