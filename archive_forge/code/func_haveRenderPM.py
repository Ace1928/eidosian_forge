import reportlab
import sys, os, fnmatch, re, functools
from configparser import ConfigParser
import unittest
from reportlab.lib.utils import isCompactDistro, __rl_loader__, rl_isdir, asUnicode
def haveRenderPM():
    from reportlab.graphics.renderPM import _getPMBackend, RenderPMError
    try:
        return _getPMBackend()
    except RenderPMError:
        return False