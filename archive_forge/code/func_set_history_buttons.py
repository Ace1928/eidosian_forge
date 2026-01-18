import uuid
import weakref
from contextlib import contextmanager
import logging
import math
import os.path
import pathlib
import sys
import tkinter as tk
import tkinter.filedialog
import tkinter.font
import tkinter.messagebox
from tkinter.simpledialog import SimpleDialog
import numpy as np
from PIL import Image, ImageTk
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook, _c_internal_utils
from matplotlib.backend_bases import (
from matplotlib._pylab_helpers import Gcf
from . import _tkagg
def set_history_buttons(self):
    state_map = {True: tk.NORMAL, False: tk.DISABLED}
    can_back = self._nav_stack._pos > 0
    can_forward = self._nav_stack._pos < len(self._nav_stack) - 1
    if 'Back' in self._buttons:
        self._buttons['Back']['state'] = state_map[can_back]
    if 'Forward' in self._buttons:
        self._buttons['Forward']['state'] = state_map[can_forward]