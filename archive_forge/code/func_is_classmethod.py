import inspect
import sys
def is_classmethod(instancemethod, klass):
    """ Determine if an instancemethod is a classmethod. """
    return inspect.ismethod(instancemethod) and instancemethod.__self__ is klass