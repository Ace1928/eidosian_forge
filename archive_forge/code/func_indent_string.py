import ast
from .qt import ClassFlag, qt_class_flags
def indent_string(self, string):
    """Start a new line by a string"""
    self._output_file.write(self._indentation)
    self._output_file.write(string)