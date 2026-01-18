import ast
from .qt import ClassFlag, qt_class_flags
def indent_line(self, line):
    """Write an indented line"""
    self._output_file.write(self._indentation)
    self._output_file.write(line)
    self._output_file.write('\n')