from .error import MarkedYAMLError
from .tokens import *
from .events import *
from .scanner import *
def dispose(self):
    self.states = []
    self.state = None