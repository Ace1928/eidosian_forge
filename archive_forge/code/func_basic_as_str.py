import traceback
from io import StringIO
from java.lang import StringBuffer  # @UnresolvedImport
from java.lang import String  # @UnresolvedImport
import java.lang  # @UnresolvedImport
import sys
from _pydev_bundle._pydev_tipper_common import do_find
from org.python.core import PyReflectedFunction  # @UnresolvedImport
from org.python import core  # @UnresolvedImport
from org.python.core import PyClass  # @UnresolvedImport
import java.util
def basic_as_str(self):
    """@returns this class information as a string (just basic format)
        """
    args = self.args
    s = 'function:%s args=%s, varargs=%s, kwargs=%s, docs:%s' % (self.name, args, self.varargs, self.kwargs, self.doc)
    return s