from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import importlib
import os
import sys
import fire
def import_from_module_name(module_name):
    """Imports a module and returns it and its name."""
    module = importlib.import_module(module_name)
    return (module, module_name)