from pyparsing import *
from sys import stdin, argv, exit
def free_if_register(self, index):
    """If index is a working register, free it, otherwise just return (helper function)"""
    if index < 0 or index > SharedData.FUNCTION_REGISTER:
        return
    else:
        self.free_register(index)