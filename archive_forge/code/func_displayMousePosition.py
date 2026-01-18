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
def displayMousePosition(xOffset=0, yOffset=0):
    """This function is meant to be run from the command line. It will
    automatically display the location and RGB of the mouse cursor."""
    try:
        runningIDLE = sys.stdin.__module__.startswith('idlelib')
    except AttributeError:
        runningIDLE = False
    print('Press Ctrl-C to quit.')
    if xOffset != 0 or yOffset != 0:
        print('xOffset: %s yOffset: %s' % (xOffset, yOffset))
    try:
        while True:
            x, y = position()
            positionStr = 'X: ' + str(x - xOffset).rjust(4) + ' Y: ' + str(y - yOffset).rjust(4)
            if not onScreen(x - xOffset, y - yOffset) or sys.platform == 'darwin':
                pixelColor = ('NaN', 'NaN', 'NaN')
            else:
                pixelColor = pyscreeze.screenshot().getpixel((x, y))
            positionStr += ' RGB: (' + str(pixelColor[0]).rjust(3)
            positionStr += ', ' + str(pixelColor[1]).rjust(3)
            positionStr += ', ' + str(pixelColor[2]).rjust(3) + ')'
            sys.stdout.write(positionStr)
            if not runningIDLE:
                sys.stdout.write('\x08' * len(positionStr))
            else:
                sys.stdout.write('\n')
                time.sleep(1)
            sys.stdout.flush()
    except KeyboardInterrupt:
        sys.stdout.write('\n')
        sys.stdout.flush()