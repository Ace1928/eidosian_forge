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
def test_key_value_pairs_to_dict(self):
    kv_list = ['str=foo', 'int=1', 'bool=true', 'list=[1, 2, 3]', 'dict={"foo": "bar"}']
    d = utils.key_value_pairs_to_dict(kv_list)
    self.assertEqual({'str': 'foo', 'int': 1, 'bool': True, 'list': [1, 2, 3], 'dict': {'foo': 'bar'}}, d)