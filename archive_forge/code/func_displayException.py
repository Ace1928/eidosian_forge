import code, sys, traceback
from ..Qt import QtWidgets, QtGui, QtCore
from ..functions import mkBrush
from .CmdInput import CmdInput
def displayException(self):
    """
        Display the current exception and stack.
        """
    tb = traceback.format_exc()
    lines = []
    indent = 4
    prefix = ''
    for l in tb.split('\n'):
        lines.append(' ' * indent + prefix + l)
    self.write('\n'.join(lines))