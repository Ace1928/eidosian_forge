from __future__ import absolute_import, division, print_function
import collections
import sys
import time
import datetime
import os
import platform
import re
import functools
from contextlib import contextmanager
@_genericPyAutoGUIChecks
def dragTo(x=None, y=None, duration=0.0, tween=linear, button=PRIMARY, logScreenshot=None, _pause=True, mouseDownUp=True):
    """Performs a mouse drag (mouse movement while a button is held down) to a
    point on the screen.

    The x and y parameters detail where the mouse event happens. If None, the
    current mouse position is used. If a float value, it is rounded down. If
    outside the boundaries of the screen, the event happens at edge of the
    screen.

    Args:
      x (int, float, None, tuple, optional): How far left (for negative values) or
        right (for positive values) to move the cursor. 0 by default. If tuple, this is used for x and y.
        If x is a str, it's considered a filename of an image to find on
        the screen with locateOnScreen() and click the center of.
      y (int, float, None, optional): How far up (for negative values) or
        down (for positive values) to move the cursor. 0 by default.
      duration (float, optional): The amount of time it takes to move the mouse
        cursor to the new xy coordinates. If 0, then the mouse cursor is moved
        instantaneously. 0.0 by default.
      tween (func, optional): The tweening function used if the duration is not
        0. A linear tween is used by default.
      button (str, int, optional): The mouse button released. TODO
      mouseDownUp (True, False): When true, the mouseUp/Down actions are not performed.
        Which allows dragging over multiple (small) actions. 'True' by default.

    Returns:
      None
    """
    x, y = _normalizeXYArgs(x, y)
    _logScreenshot(logScreenshot, 'dragTo', '%s,%s' % (x, y), folder='.')
    if mouseDownUp:
        mouseDown(button=button, logScreenshot=False, _pause=False)
    _mouseMoveDrag('drag', x, y, 0, 0, duration, tween, button)
    if mouseDownUp:
        mouseUp(button=button, logScreenshot=False, _pause=False)