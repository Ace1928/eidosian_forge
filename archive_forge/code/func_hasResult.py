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
def hasResult(self):
    """Returns True if the result for this request has arrived."""
    try:
        self.result(block=False)
    except NoResultError:
        pass
    return self.gotResult