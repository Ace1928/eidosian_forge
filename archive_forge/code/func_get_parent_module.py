import inspect
import random
import re
import unittest
from . import import_submodule
def get_parent_module(self, class_):
    if class_ not in self.parent_modules:
        self.parent_modules[class_] = import_submodule(class_.__module__)
    return self.parent_modules[class_]