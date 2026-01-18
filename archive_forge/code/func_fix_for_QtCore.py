import is finished. But that is no problem since the module is passed in.
import warnings
from textwrap import dedent
def fix_for_QtCore(QtCore):
    from enum import Flag
    Qt = QtCore.Qt
    flag_or = Flag.__or__

    def func_or(self, other):
        if isinstance(self, Flag) and isinstance(other, Flag):
            return Qt.KeyboardModifier(self.value | other.value)
        return QtCore.QKeyCombination(self, other)

    def func_add(self, other):
        warnings.warn(dedent(f'\n            The "+" operator is deprecated in Qt For Python 6.0 .\n            Please use "|" instead.'), PySideDeprecationWarningRemovedInQt6, stacklevel=2)
        return func_or(self, other)
    Qt.KeyboardModifier.__or__ = func_or
    Qt.KeyboardModifier.__ror__ = func_or
    Qt.Modifier.__or__ = func_or
    Qt.Modifier.__ror__ = func_or
    Qt.KeyboardModifier.__add__ = func_add
    Qt.KeyboardModifier.__radd__ = func_add
    Qt.Modifier.__add__ = func_add
    Qt.Modifier.__radd__ = func_add