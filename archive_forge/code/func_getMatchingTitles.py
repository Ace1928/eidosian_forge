import keyword
import os
import pkgutil
import re
import subprocess
import sys
from argparse import Namespace
from collections import OrderedDict
from functools import lru_cache
import pyqtgraph as pg
from pyqtgraph.Qt import QT_LIB, QtCore, QtGui, QtWidgets
import exampleLoaderTemplate_generic as ui_template
import utils
def getMatchingTitles(self, text, exDict=None, acceptAll=False):
    if exDict is None:
        exDict = utils.examples_
    text = text.lower()
    titles = []
    for kk, vv in exDict.items():
        matched = acceptAll or text in kk.lower()
        if isinstance(vv, dict):
            titles.extend(self.getMatchingTitles(text, vv, acceptAll=matched))
        elif matched:
            titles.append(kk)
    return titles