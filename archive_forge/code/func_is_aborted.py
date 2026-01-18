from pynvml.nvml import *
import datetime
import collections
import time
from threading import Thread
def is_aborted(self):
    return self.__abort