import functools
import dill
import sys
def function_with_unassigned_variable():
    if False:
        value = None
    return lambda: value