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
def test_convert_list_props_to_comma_separated_strings(self):
    data = {'prop1': 'val1', 'prop2': ['item1', 'item2', 'item3']}
    result = utils.convert_list_props_to_comma_separated(data)
    self.assertEqual('val1', result['prop1'])
    self.assertEqual('item1, item2, item3', result['prop2'])