import collections
import datetime
import functools
import os
import subprocess
import sys
import time
import errno
from contextlib import contextmanager
from PIL import Image
from PIL import ImageOps
from PIL import ImageDraw
from PIL import __version__ as PIL__version__
from PIL import ImageGrab
def locateOnScreenNear(image, x, y):
    """
    TODO
    """
    foundMatchesBoxes = list(locateAllOnScreen(image))
    distancesSquared = []
    shortestDistanceIndex = 0
    for foundMatchesBox in foundMatchesBoxes:
        foundMatchX, foundMatchY = center(foundMatchesBox)
        xDistance = abs(x - foundMatchX)
        yDistance = abs(y - foundMatchY)
        distancesSquared.append(xDistance * xDistance + yDistance * yDistance)
        if distancesSquared[-1] < distancesSquared[shortestDistanceIndex]:
            shortestDistanceIndex = len(distancesSquared) - 1
    return foundMatchesBoxes[shortestDistanceIndex]