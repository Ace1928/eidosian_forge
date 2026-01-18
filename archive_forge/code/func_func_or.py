import is finished. But that is no problem since the module is passed in.
import warnings
from textwrap import dedent
def func_or(self, other):
    if isinstance(self, Flag) and isinstance(other, Flag):
        return Qt.KeyboardModifier(self.value | other.value)
    return QtCore.QKeyCombination(self, other)