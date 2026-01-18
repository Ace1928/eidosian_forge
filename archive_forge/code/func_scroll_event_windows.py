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
def scroll_event_windows(event):
    self = weakself()
    if self is None:
        root = weakroot()
        if root is not None:
            root.unbind('<MouseWheel>', scroll_event_windows_id)
        return
    return self.scroll_event_windows(event)