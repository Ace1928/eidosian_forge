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
@classmethod
def registerObject(cls, obj):
    pid = cls.nextProxyId
    cls.nextProxyId += 1
    cls.proxiedObjects[pid] = obj
    return pid