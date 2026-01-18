import re
import warnings
import weakref
from collections import OrderedDict
from .. import functions as fn
from ..Qt import QtCore
from .ParameterItem import ParameterItem
def emitTreeChanges(self):
    if self.blockTreeChangeEmit == 0:
        changes = self.treeStateChanges
        self.treeStateChanges = []
        if len(changes) > 0:
            self.sigTreeStateChanged.emit(self, changes)