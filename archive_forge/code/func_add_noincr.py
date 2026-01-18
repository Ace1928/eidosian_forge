from .abstract import Thenable
from .promises import promise
def add_noincr(self, p):
    if not self.cancelled:
        if self.ready:
            raise ValueError('Cannot add promise to full barrier')
        p.then(self)