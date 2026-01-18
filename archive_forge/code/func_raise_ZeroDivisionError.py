import inspect
from .. import decorators, lock
from . import TestCase
def raise_ZeroDivisionError(self):
    1 / 0