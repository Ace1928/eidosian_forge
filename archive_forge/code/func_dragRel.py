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
def dragRel(xOffset=0, yOffset=0, duration=0.0, tween=linear, button=PRIMARY, logScreenshot=None, _pause=True, mouseDownUp=True):
    """Performs a mouse drag (mouse movement while a button is held down) to a
    point on the screen, relative to its current position.

    The x and y parameters detail where the mouse event happens. If None, the
    current mouse position is used. If a float value, it is rounded down. If
    outside the boundaries of the screen, the event happens at edge of the
    screen.

    Args:
      x (int, float, None, tuple, optional): How far left (for negative values) or
        right (for positive values) to move the cursor. 0 by default. If tuple, this is used for xOffset and yOffset.
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
    if xOffset is None:
        xOffset = 0
    if yOffset is None:
        yOffset = 0
    if type(xOffset) in (tuple, list):
        xOffset, yOffset = (xOffset[0], xOffset[1])
    if xOffset == 0 and yOffset == 0:
        return
    mousex, mousey = platformModule._position()
    _logScreenshot(logScreenshot, 'dragRel', '%s,%s' % (xOffset, yOffset), folder='.')
    if mouseDownUp:
        mouseDown(button=button, logScreenshot=False, _pause=False)
    _mouseMoveDrag('drag', mousex, mousey, xOffset, yOffset, duration, tween, button)
    if mouseDownUp:
        mouseUp(button=button, logScreenshot=False, _pause=False)