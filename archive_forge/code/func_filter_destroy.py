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
def filter_destroy(event):
    self = weakself()
    if self is None:
        root = weakroot()
        if root is not None:
            root.unbind('<Destroy>', filter_destroy_id)
        return
    if event.widget is self._tkcanvas:
        CloseEvent('close_event', self)._process()