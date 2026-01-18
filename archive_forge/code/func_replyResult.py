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
def replyResult(self, reqId, result):
    self.send(request='result', reqId=reqId, callSync='off', opts=dict(result=result))