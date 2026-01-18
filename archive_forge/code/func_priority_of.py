from sympy.external import import_module
import os
def priority_of(self, op):
    """To get the priority of given operator"""
    if op in ['=', '+=', '-=', '*=', '/=', '%=']:
        return 1
    if op in ['&&', '||']:
        return 2
    if op in ['<', '<=', '>', '>=', '==', '!=']:
        return 3
    if op in ['+', '-']:
        return 4
    if op in ['*', '/', '%']:
        return 5
    return 0