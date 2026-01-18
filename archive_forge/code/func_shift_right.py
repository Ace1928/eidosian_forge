import types
from _pydevd_bundle.pydevd_constants import IS_JYTHON
from _pydev_bundle._pydev_imports_tipper import signature_from_docstring
def shift_right(string, prefix):
    return ''.join((prefix + line for line in string.splitlines(True)))