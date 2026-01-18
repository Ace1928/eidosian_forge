import operator
import warnings
def presence(label):
    """Higher-order function to test presence of a given label"""
    return lambda x, y: 1.0 * ((label in x) == (label in y))