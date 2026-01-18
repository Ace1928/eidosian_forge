import posixpath
import re
from ._hl.attrs import AttributeManager
from ._hl.base import HLObject
from IPython import get_ipython
from IPython.core.error import TryNext
from IPython.utils import generics
def h5py_item_completer(context, command):
    """Compute possible item matches for dict-like objects"""
    base, item = re_item_match.split(command)[1:4:2]
    try:
        obj = _retrieve_obj(base, context)
    except Exception:
        return []
    path, _ = posixpath.split(item)
    try:
        if path:
            items = (posixpath.join(path, name) for name in obj[path].keys())
        else:
            items = obj.keys()
    except AttributeError:
        return []
    items = list(items)
    return [i for i in items if i[:len(item)] == item]