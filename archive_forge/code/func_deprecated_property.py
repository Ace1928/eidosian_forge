import re
from datetime import (
from collections import namedtuple
from webob.byterange import (
from webob.compat import (
from webob.datetime_utils import (
from webob.util import (
def deprecated_property(attr, name, text, version):
    """
    Wraps a descriptor, with a deprecation warning or error
    """

    def warn():
        warn_deprecation('The attribute %s is deprecated: %s' % (name, text), version, 3)

    def fget(self):
        warn()
        return attr.__get__(self, type(self))

    def fset(self, val):
        warn()
        attr.__set__(self, val)

    def fdel(self):
        warn()
        attr.__delete__(self)
    return property(fget, fset, fdel, '<Deprecated attribute %s>' % name)