from .error import MarkedYAMLError
from .tokens import *
from .events import *
from .scanner import *
def process_empty_scalar(self, mark):
    return ScalarEvent(None, None, (True, False), '', mark, mark)