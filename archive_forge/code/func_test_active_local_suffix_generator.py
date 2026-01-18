import os
import itertools
import logging
import pickle
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.core.base.suffix import (
from pyomo.environ import (
from io import StringIO
def test_active_local_suffix_generator(self):
    model = ConcreteModel()
    model.junk_LOCAL_int = Suffix(direction=Suffix.LOCAL, datatype=Suffix.INT)
    model.junk_LOCAL_float = Suffix(direction=Suffix.LOCAL, datatype=Suffix.FLOAT)
    model.junk_IMPORT_EXPORT = Suffix(direction=Suffix.IMPORT_EXPORT, datatype=None)
    model.junk_EXPORT = Suffix(direction=Suffix.EXPORT, datatype=None)
    model.junk_IMPORT = Suffix(direction=Suffix.IMPORT, datatype=None)
    suffixes = dict(active_local_suffix_generator(model))
    self.assertTrue('junk_LOCAL_int' in suffixes)
    self.assertTrue('junk_LOCAL_float' in suffixes)
    self.assertTrue('junk_IMPORT_EXPORT' not in suffixes)
    self.assertTrue('junk_EXPORT' not in suffixes)
    self.assertTrue('junk_IMPORT' not in suffixes)
    model.junk_LOCAL_float.deactivate()
    suffixes = dict(active_local_suffix_generator(model))
    self.assertTrue('junk_LOCAL_int' in suffixes)
    self.assertTrue('junk_LOCAL_float' not in suffixes)
    self.assertTrue('junk_IMPORT_EXPORT' not in suffixes)
    self.assertTrue('junk_EXPORT' not in suffixes)
    self.assertTrue('junk_IMPORT' not in suffixes)
    model.junk_LOCAL_float.activate()
    suffixes = dict(active_local_suffix_generator(model, datatype=Suffix.FLOAT))
    self.assertTrue('junk_LOCAL_int' not in suffixes)
    self.assertTrue('junk_LOCAL_float' in suffixes)
    self.assertTrue('junk_IMPORT_EXPORT' not in suffixes)
    self.assertTrue('junk_EXPORT' not in suffixes)
    self.assertTrue('junk_IMPORT' not in suffixes)
    model.junk_LOCAL_float.deactivate()
    suffixes = dict(active_local_suffix_generator(model, datatype=Suffix.FLOAT))
    self.assertTrue('junk_LOCAL_int' not in suffixes)
    self.assertTrue('junk_LOCAL_float' not in suffixes)
    self.assertTrue('junk_IMPORT_EXPORT' not in suffixes)
    self.assertTrue('junk_EXPORT' not in suffixes)
    self.assertTrue('junk_IMPORT' not in suffixes)