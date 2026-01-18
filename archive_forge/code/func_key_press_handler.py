from collections import namedtuple
from contextlib import ExitStack, contextmanager, nullcontext
from enum import Enum, IntEnum
import functools
import importlib
import inspect
import io
import itertools
import logging
import os
import sys
import time
import weakref
from weakref import WeakKeyDictionary
import numpy as np
import matplotlib as mpl
from matplotlib import (
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_managers import ToolManager
from matplotlib.cbook import _setattr_cm
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
from matplotlib._enums import JoinStyle, CapStyle
def key_press_handler(event, canvas=None, toolbar=None):
    """
    Implement the default Matplotlib key bindings for the canvas and toolbar
    described at :ref:`key-event-handling`.

    Parameters
    ----------
    event : `KeyEvent`
        A key press/release event.
    canvas : `FigureCanvasBase`, default: ``event.canvas``
        The backend-specific canvas instance.  This parameter is kept for
        back-compatibility, but, if set, should always be equal to
        ``event.canvas``.
    toolbar : `NavigationToolbar2`, default: ``event.canvas.toolbar``
        The navigation cursor toolbar.  This parameter is kept for
        back-compatibility, but, if set, should always be equal to
        ``event.canvas.toolbar``.
    """
    if event.key is None:
        return
    if canvas is None:
        canvas = event.canvas
    if toolbar is None:
        toolbar = canvas.toolbar
    fullscreen_keys = rcParams['keymap.fullscreen']
    home_keys = rcParams['keymap.home']
    back_keys = rcParams['keymap.back']
    forward_keys = rcParams['keymap.forward']
    pan_keys = rcParams['keymap.pan']
    zoom_keys = rcParams['keymap.zoom']
    save_keys = rcParams['keymap.save']
    quit_keys = rcParams['keymap.quit']
    quit_all_keys = rcParams['keymap.quit_all']
    grid_keys = rcParams['keymap.grid']
    grid_minor_keys = rcParams['keymap.grid_minor']
    toggle_yscale_keys = rcParams['keymap.yscale']
    toggle_xscale_keys = rcParams['keymap.xscale']
    if event.key in fullscreen_keys:
        try:
            canvas.manager.full_screen_toggle()
        except AttributeError:
            pass
    if event.key in quit_keys:
        Gcf.destroy_fig(canvas.figure)
    if event.key in quit_all_keys:
        Gcf.destroy_all()
    if toolbar is not None:
        if event.key in home_keys:
            toolbar.home()
        elif event.key in back_keys:
            toolbar.back()
        elif event.key in forward_keys:
            toolbar.forward()
        elif event.key in pan_keys:
            toolbar.pan()
            toolbar._update_cursor(event)
        elif event.key in zoom_keys:
            toolbar.zoom()
            toolbar._update_cursor(event)
        elif event.key in save_keys:
            toolbar.save_figure()
    if event.inaxes is None:
        return

    def _get_uniform_gridstate(ticks):
        if all((tick.gridline.get_visible() for tick in ticks)):
            return True
        elif not any((tick.gridline.get_visible() for tick in ticks)):
            return False
        else:
            return None
    ax = event.inaxes
    if event.key in grid_keys and None not in [_get_uniform_gridstate(ax.xaxis.minorTicks), _get_uniform_gridstate(ax.yaxis.minorTicks)]:
        x_state = _get_uniform_gridstate(ax.xaxis.majorTicks)
        y_state = _get_uniform_gridstate(ax.yaxis.majorTicks)
        cycle = [(False, False), (True, False), (True, True), (False, True)]
        try:
            x_state, y_state = cycle[(cycle.index((x_state, y_state)) + 1) % len(cycle)]
        except ValueError:
            pass
        else:
            ax.grid(x_state, which='major' if x_state else 'both', axis='x')
            ax.grid(y_state, which='major' if y_state else 'both', axis='y')
            canvas.draw_idle()
    if event.key in grid_minor_keys and None not in [_get_uniform_gridstate(ax.xaxis.majorTicks), _get_uniform_gridstate(ax.yaxis.majorTicks)]:
        x_state = _get_uniform_gridstate(ax.xaxis.minorTicks)
        y_state = _get_uniform_gridstate(ax.yaxis.minorTicks)
        cycle = [(False, False), (True, False), (True, True), (False, True)]
        try:
            x_state, y_state = cycle[(cycle.index((x_state, y_state)) + 1) % len(cycle)]
        except ValueError:
            pass
        else:
            ax.grid(x_state, which='both', axis='x')
            ax.grid(y_state, which='both', axis='y')
            canvas.draw_idle()
    elif event.key in toggle_yscale_keys:
        scale = ax.get_yscale()
        if scale == 'log':
            ax.set_yscale('linear')
            ax.figure.canvas.draw_idle()
        elif scale == 'linear':
            try:
                ax.set_yscale('log')
            except ValueError as exc:
                _log.warning(str(exc))
                ax.set_yscale('linear')
            ax.figure.canvas.draw_idle()
    elif event.key in toggle_xscale_keys:
        scalex = ax.get_xscale()
        if scalex == 'log':
            ax.set_xscale('linear')
            ax.figure.canvas.draw_idle()
        elif scalex == 'linear':
            try:
                ax.set_xscale('log')
            except ValueError as exc:
                _log.warning(str(exc))
                ax.set_xscale('linear')
            ax.figure.canvas.draw_idle()