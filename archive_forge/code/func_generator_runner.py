from io import StringIO
import os
import sys
import types
import json
from copy import deepcopy
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.base.block import (
import pyomo.core.expr as EXPR
from pyomo.opt import check_available_solvers
from pyomo.gdp import Disjunct
def generator_runner(self, ctype):
    model = self.generate_model()
    for block in model.block_data_objects(sort=SortComponents.indices):
        generator = None
        try:
            generator = list(block.component_objects(ctype, active=True, descend_into=False))
        except:
            if issubclass(ctype, Component):
                print('component_objects(active=True) failed with ctype %s' % ctype)
                raise
        else:
            if not issubclass(ctype, Component):
                self.fail('component_objects(active=True) should have failed with ctype %s' % ctype)
            self.assertEqual([comp.name for comp in generator], [comp.name for comp in block.component_lists[ctype]])
            self.assertEqual([id(comp) for comp in generator], [id(comp) for comp in block.component_lists[ctype]])
        generator = None
        try:
            generator = list(block.component_objects(ctype, descend_into=False))
        except:
            if issubclass(ctype, Component):
                print('components failed with ctype %s' % ctype)
                raise
        else:
            if not issubclass(ctype, Component):
                self.fail('components should have failed with ctype %s' % ctype)
            self.assertEqual([comp.name for comp in generator], [comp.name for comp in block.component_lists[ctype]])
            self.assertEqual([id(comp) for comp in generator], [id(comp) for comp in block.component_lists[ctype]])
        generator = None
        try:
            generator = list(block.component_data_iterindex(ctype, active=True, sort=False, descend_into=False))
        except:
            if issubclass(ctype, Component):
                print('component_data_objects(active=True, sort_by_keys=False) failed with ctype %s' % ctype)
                raise
        else:
            if not issubclass(ctype, Component):
                self.fail('component_data_objects(active=True, sort_by_keys=False) should have failed with ctype %s' % ctype)
            self.assertEqual([comp.name for name, comp in generator], [comp.name for comp in block.component_data_lists[ctype]])
            self.assertEqual([id(comp) for name, comp in generator], [id(comp) for comp in block.component_data_lists[ctype]])
        generator = None
        try:
            generator = list(block.component_data_iterindex(ctype, active=True, sort=True, descend_into=False))
        except:
            if issubclass(ctype, Component):
                print('component_data_objects(active=True, sort=True) failed with ctype %s' % ctype)
                raise
        else:
            if not issubclass(ctype, Component):
                self.fail('component_data_objects(active=True, sort=True) should have failed with ctype %s' % ctype)
            self.assertEqual(sorted([comp.name for name, comp in generator]), sorted([comp.name for comp in block.component_data_lists[ctype]]))
            self.assertEqual(sorted([id(comp) for name, comp in generator]), sorted([id(comp) for comp in block.component_data_lists[ctype]]))
        generator = None
        try:
            generator = list(block.component_data_iterindex(ctype, sort=False, descend_into=False))
        except:
            if issubclass(ctype, Component):
                print('components_data(sort_by_keys=True) failed with ctype %s' % ctype)
                raise
        else:
            if not issubclass(ctype, Component):
                self.fail('components_data(sort_by_keys=True) should have failed with ctype %s' % ctype)
            self.assertEqual([comp.name for name, comp in generator], [comp.name for comp in block.component_data_lists[ctype]])
            self.assertEqual([id(comp) for name, comp in generator], [id(comp) for comp in block.component_data_lists[ctype]])
        generator = None
        try:
            generator = list(block.component_data_iterindex(ctype, sort=True, descend_into=False))
        except:
            if issubclass(ctype, Component):
                print('components_data(sort_by_keys=False) failed with ctype %s' % ctype)
                raise
        else:
            if not issubclass(ctype, Component):
                self.fail('components_data(sort_by_keys=False) should have failed with ctype %s' % ctype)
            self.assertEqual(sorted([comp.name for name, comp in generator]), sorted([comp.name for comp in block.component_data_lists[ctype]]))
            self.assertEqual(sorted([id(comp) for name, comp in generator]), sorted([id(comp) for comp in block.component_data_lists[ctype]]))