from __future__ import absolute_import
import math
import struct
import dns.inet
def get_option_class(otype):
    """Return the class for the specified option type.

    The GenericOption class is used if a more specific class is not
    known.
    """
    cls = _type_to_class.get(otype)
    if cls is None:
        cls = GenericOption
    return cls