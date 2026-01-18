import sys
import warnings
from collections import deque
from functools import wraps
def preserve(self):
    return self.pop_all()