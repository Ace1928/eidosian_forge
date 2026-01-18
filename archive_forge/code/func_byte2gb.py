import gc
import psutil
import threading
import torch
from accelerate.utils import is_xpu_available
def byte2gb(x):
    return int(x / 2 ** 30)