from lxml import etree
import sys
import re
import doctest
def install_clone(self):
    self.func_code = self.check_func.__code__
    self.func_globals = self.check_func.__globals__
    self.check_func.__code__ = self.clone_func.__code__