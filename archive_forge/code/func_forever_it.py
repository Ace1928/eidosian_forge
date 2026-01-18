import string
from taskflow import test
from taskflow.utils import iter_utils
def forever_it():
    i = 0
    while True:
        yield i
        i += 1