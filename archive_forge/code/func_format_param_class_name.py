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
def format_param_class_name(paramClassName):
    if paramClassName.startswith("<type '") and paramClassName.endswith("'>"):
        paramClassName = paramClassName[len("<type '"):-2]
    if paramClassName.startswith('['):
        if paramClassName == '[C':
            paramClassName = 'char[]'
        elif paramClassName == '[B':
            paramClassName = 'byte[]'
        elif paramClassName == '[I':
            paramClassName = 'int[]'
        elif paramClassName.startswith('[L') and paramClassName.endswith(';'):
            paramClassName = paramClassName[2:-1]
            paramClassName += '[]'
    return paramClassName