import inspect
import pytest
from ..utils import deprecation
from .utils import call_method
def level2():
    deprecation('test message', [])