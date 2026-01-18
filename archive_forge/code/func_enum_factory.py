import importlib.abc
import sys
import os
import types
from functools import partial, lru_cache
import operator
def enum_factory(QT_API, QtCore):
    """Construct an enum helper to account for PyQt5 <-> PyQt6 changes."""

    @lru_cache(None)
    def _enum(name):
        return operator.attrgetter(name if QT_API == QT_API_PYQT6 else name.rpartition('.')[0])(sys.modules[QtCore.__package__])
    return _enum