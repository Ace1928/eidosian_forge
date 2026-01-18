import os
import sys
import threading
import time
import traceback
import warnings
import weakref
import builtins
import pickle
import numpy as np
from ..util import cprint
def autoProxy(self, obj, noProxyTypes):
    for typ in noProxyTypes:
        if isinstance(obj, typ):
            return obj
    return LocalObjectProxy(obj)