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
def get_as_doc(self):
    s = str(self.name)
    if self.doc:
        s += '\n@doc %s\n' % str(self.doc)
    if self.args:
        s += '\n@params '
        for arg in self.args:
            s += str(format_param_class_name(arg))
            s += '  '
    if self.varargs:
        s += '\n@varargs '
        s += str(self.varargs)
    if self.kwargs:
        s += '\n@kwargs '
        s += str(self.kwargs)
    if self.ret:
        s += '\n@return '
        s += str(format_param_class_name(str(self.ret)))
    return str(s)