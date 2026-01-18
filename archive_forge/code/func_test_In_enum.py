import argparse
import enum
import os
import os.path
import pickle
import re
import sys
import types
import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
from pyomo.common.config import (
from pyomo.common.log import LoggingIntercept
def test_In_enum(self):

    class TestEnum(enum.Enum):
        ITEM_ONE = 1
        ITEM_TWO = 'two'
    cfg = ConfigDict()
    cfg.declare('enum', ConfigValue(default=TestEnum.ITEM_TWO, domain=In(TestEnum)))
    self.assertEqual(cfg.get('enum').domain_name(), 'InEnum[TestEnum]')
    self.assertEqual(cfg.enum, TestEnum.ITEM_TWO)
    cfg.enum = 'ITEM_ONE'
    self.assertEqual(cfg.enum, TestEnum.ITEM_ONE)
    cfg.enum = TestEnum.ITEM_TWO
    self.assertEqual(cfg.enum, TestEnum.ITEM_TWO)
    cfg.enum = 1
    self.assertEqual(cfg.enum, TestEnum.ITEM_ONE)
    cfg.enum = 'two'
    self.assertEqual(cfg.enum, TestEnum.ITEM_TWO)
    with self.assertRaisesRegex(ValueError, '.*3 is not a valid'):
        cfg.enum = 3
    with self.assertRaisesRegex(ValueError, '.*invalid value'):
        cfg.enum = 'ITEM_THREE'