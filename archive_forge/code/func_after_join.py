import collections
import functools
import threading
import time
from taskflow import test
from taskflow.utils import threading_utils as tu
def after_join(t):
    events.append('aj')