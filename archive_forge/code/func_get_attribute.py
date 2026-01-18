from pyparsing import *
from sys import stdin, argv, exit
def get_attribute(self, index):
    try:
        return self.table[index].attribute
    except Exception:
        self.error()