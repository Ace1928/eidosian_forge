import functools
import warnings
from collections import namedtuple
import gi.module
from gi.overrides import override, deprecated_attr
from gi.repository import GLib
from gi import PyGIDeprecationWarning
from gi import _propertyhelper as propertyhelper
from gi import _signalhelper as signalhelper
from gi import _gi
from gi import _option as option
def signal_query(id_or_name, type_=None):
    if type_ is not None:
        id_or_name = signal_lookup(id_or_name, type_)
    res = GObjectModule.signal_query(id_or_name)
    assert res is not None
    if res.signal_id == 0:
        return None
    return SignalQuery(res.signal_id, res.signal_name, res.itype, res.signal_flags, res.return_type, tuple(res.param_types))