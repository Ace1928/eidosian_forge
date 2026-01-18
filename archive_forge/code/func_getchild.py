from suds import *
from suds.sax import splitPrefix, Namespace
from suds.sudsobject import Object
from suds.xsd.query import BlindQuery, TypeQuery, qualify
import re
from logging import getLogger
def getchild(self, name, parent):
    """Get a child by name."""
    log.debug('searching parent (%s) for (%s)', Repr(parent), name)
    if name.startswith('@'):
        return parent.get_attribute(name[1:])
    return parent.get_child(name)