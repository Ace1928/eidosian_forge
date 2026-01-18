from pyparsing import *
from sys import stdin, argv, exit
def same_type_as_argument(self, index, function_index, argument_number):
    """Returns True if index and function's argument are of the same type
           index - index in symbol table
           function_index - function's index in symbol table
           argument_number - # of function's argument
        """
    try:
        same = self.table[function_index].param_types[argument_number] == self.table[index].type
    except Exception:
        self.error()
    return same