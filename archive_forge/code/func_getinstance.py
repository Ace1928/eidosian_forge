import functools
import warnings
import threading
import sys
def getinstance():
    if cls not in instances:
        instances[cls] = cls()
    return instances[cls]