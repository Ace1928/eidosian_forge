from concurrent.futures import ThreadPoolExecutor
from promise import Promise
import time
import weakref
import gc
def my(resolve, reject):
    if x > 0:
        resolve(x)
    else:
        reject(Exception(x))