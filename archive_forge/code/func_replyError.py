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
def replyError(self, reqId, *exc):
    excStr = traceback.format_exception(*exc)
    try:
        self.send(request='error', reqId=reqId, callSync='off', opts=dict(exception=exc[1], excString=excStr))
    except:
        self.send(request='error', reqId=reqId, callSync='off', opts=dict(exception=None, excString=excStr))