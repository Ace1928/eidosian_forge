import os
import sys
import pickle
import subprocess
from .. import getConfigOption
from ..Qt import QtCore, QtWidgets
from .repl_widget import ReplWidget
from .exception_widget import ExceptionHandlerWidget
def saveHistory(self, history):
    """Store the list of previously-invoked command strings."""
    if self.historyFile is not None:
        with open(self.historyFile, 'wb') as pf:
            pickle.dump(history, pf)