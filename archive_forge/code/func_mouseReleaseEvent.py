import functools
import os
import sys
import traceback
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
import matplotlib.backends.qt_editor.figureoptions as figureoptions
from . import qt_compat
from .qt_compat import (
def mouseReleaseEvent(self, event):
    button = self.buttond.get(event.button())
    if button is not None:
        MouseEvent('button_release_event', self, *self.mouseEventCoords(event), button, modifiers=self._mpl_modifiers(), guiEvent=event)._process()