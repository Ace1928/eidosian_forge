import os
import re
import subprocess
import sys
import time
import warnings
from . import QtCore, QtGui, QtWidgets, compat
from . import internals
def isQObjectAlive(obj):
    return not sip.isdeleted(obj)