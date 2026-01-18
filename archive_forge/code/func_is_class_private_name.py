import inspect
import sys
def is_class_private_name(name):
    """ Determine if a name is a class private name. """
    return name.startswith('__') and (not name.endswith('__'))