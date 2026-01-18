from pyparsing import *
from sys import stdin, argv, exit
def attribute_str(self):
    """Returns attribute string (used only for table display)"""
    return '{0}={1}'.format(self.attribute_name, self.attribute) if self.attribute != None else 'None'