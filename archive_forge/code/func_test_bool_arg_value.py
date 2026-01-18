import builtins
import json
import os
import subprocess
import sys
import tempfile
from unittest import mock
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.tests.unit import utils as test_utils
def test_bool_arg_value(self):
    self.assertTrue(utils.bool_argument_value('arg', 'y', strict=True))
    self.assertTrue(utils.bool_argument_value('arg', 'TrUe', strict=True))
    self.assertTrue(utils.bool_argument_value('arg', '1', strict=True))
    self.assertTrue(utils.bool_argument_value('arg', 1, strict=True))
    self.assertFalse(utils.bool_argument_value('arg', '0', strict=True))
    self.assertFalse(utils.bool_argument_value('arg', 'No', strict=True))
    self.assertRaises(exc.CommandError, utils.bool_argument_value, 'arg', 'ee', strict=True)
    self.assertFalse(utils.bool_argument_value('arg', 'ee', strict=False))
    self.assertTrue(utils.bool_argument_value('arg', 'ee', strict=False, default=True))
    self.assertEqual('foo', utils.bool_argument_value('arg', 'ee', strict=False, default='foo'))