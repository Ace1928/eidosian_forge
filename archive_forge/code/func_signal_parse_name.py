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
def signal_parse_name(detailed_signal, itype, force_detail_quark):
    """Parse a detailed signal name into (signal_id, detail).

    :param str detailed_signal:
        Signal name which can include detail.
        For example: "notify:prop_name"
    :returns:
        Tuple of (signal_id, detail)
    :raises ValueError:
        If the given signal is unknown.
    """
    res, signal_id, detail = GObjectModule.signal_parse_name(detailed_signal, itype, force_detail_quark)
    if res:
        return (signal_id, detail)
    else:
        raise ValueError('%s: unknown signal name: %s' % (itype, detailed_signal))