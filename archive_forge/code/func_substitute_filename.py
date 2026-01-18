import sys
import os
import re
import pkg_resources
from string import Template
def substitute_filename(fn, variables):
    """ Substitute +variables+ in file directory names. """
    for var, value in variables.items():
        fn = fn.replace('+%s+' % var, str(value))
    return fn