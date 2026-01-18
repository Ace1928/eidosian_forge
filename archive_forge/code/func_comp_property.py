import sys
import logging
import os.path
import re
from pyside2uic.exceptions import NoSuchWidgetError
from pyside2uic.objcreator import QObjectCreator
from pyside2uic.properties import Properties
def comp_property(m, so_far=-2, nr=0):
    if m >= 0:
        nr += 1
        if so_far == -2:
            so_far = m
        elif so_far != m:
            so_far = -1
    return (so_far, nr)