import sys
from . import server
from .workers import threadpool
from ._compat import ntob, bton
@numthreads.setter
def numthreads(self, value):
    self.requests.min = value