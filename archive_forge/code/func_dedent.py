import ast
from .qt import ClassFlag, qt_class_flags
def dedent(self):
    """Decrease indentation level"""
    self._indent_level = self._indent_level - 1
    self._indentation = '    ' * self._indent_level