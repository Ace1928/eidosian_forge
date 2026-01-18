import is finished. But that is no problem since the module is passed in.
import warnings
from textwrap import dedent
def func_add(self, other):
    warnings.warn(dedent(f'\n            The "+" operator is deprecated in Qt For Python 6.0 .\n            Please use "|" instead.'), PySideDeprecationWarningRemovedInQt6, stacklevel=2)
    return func_or(self, other)