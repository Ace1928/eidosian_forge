import logging
import warnings
from passlib import hash
from passlib.utils.compat import u
from passlib.tests.utils import TestCase, HandlerCase
from passlib.tests.test_handlers import UPASS_WAV
def test_94_using_w_algs(self):
    """using() -- 'algs' parameter"""
    self.test_94_using_w_default_algs(param='algs')