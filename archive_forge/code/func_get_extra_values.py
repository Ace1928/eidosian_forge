import os
import re
import sys
from codecs import BOM_UTF8, BOM_UTF16, BOM_UTF16_BE, BOM_UTF16_LE
import six
from ._version import __version__
def get_extra_values(conf, _prepend=()):
    """
    Find all the values and sections not in the configspec from a validated
    ConfigObj.
    
    ``get_extra_values`` returns a list of tuples where each tuple represents
    either an extra section, or an extra value.
    
    The tuples contain two values, a tuple representing the section the value 
    is in and the name of the extra values. For extra values in the top level
    section the first member will be an empty tuple. For values in the 'foo'
    section the first member will be ``('foo',)``. For members in the 'bar'
    subsection of the 'foo' section the first member will be ``('foo', 'bar')``.
    
    NOTE: If you call ``get_extra_values`` on a ConfigObj instance that hasn't
    been validated it will return an empty list.
    """
    out = []
    out.extend([(_prepend, name) for name in conf.extra_values])
    for name in conf.sections:
        if name not in conf.extra_values:
            out.extend(get_extra_values(conf[name], _prepend + (name,)))
    return out