import posixpath
import re
from ._hl.attrs import AttributeManager
from ._hl.base import HLObject
from IPython import get_ipython
from IPython.core.error import TryNext
from IPython.utils import generics
def h5py_completer(self, event):
    """ Completer function to be loaded into IPython """
    base = re_object_match.split(event.line)[1]
    try:
        obj = self._ofind(base).obj
    except AttributeError:
        obj = self._ofind(base).get('obj')
    if not isinstance(obj, (AttributeManager, HLObject)):
        raise TryNext
    try:
        return h5py_attr_completer(self, event.line)
    except ValueError:
        pass
    try:
        return h5py_item_completer(self, event.line)
    except ValueError:
        pass
    return []