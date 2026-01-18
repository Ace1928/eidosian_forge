from twisted.python import usage
from twisted.trial import unittest
def opt_myparam(self, value):
    self.opts['myparam'] = f'{value} WITH A PONY!'