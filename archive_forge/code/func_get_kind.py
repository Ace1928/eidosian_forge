from pyparsing import *
from sys import stdin, argv, exit
def get_kind(self, index):
    try:
        return self.table[index].kind
    except Exception:
        self.error()