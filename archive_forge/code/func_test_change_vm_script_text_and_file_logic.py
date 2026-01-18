import re
import sys
import datetime
import unittest
import traceback
from unittest.mock import patch, mock_open
from libcloud.test import MockHttp
from libcloud.utils.py3 import ET, PY2, b, httplib, assertRaisesRegex
from libcloud.compute.base import Node, NodeImage
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import VCLOUD_PARAMS
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vcloud import (
@patch('libcloud.compute.drivers.vcloud.VCloud_1_5_NodeDriver._get_vm_elements', side_effect=CallException('Called'))
def test_change_vm_script_text_and_file_logic(self, _):
    assertion_error = False
    for vm_script_file, vm_script_text, open_succeeds, open_call_count, returned_early in ((None, None, True, 0, True), (None, None, False, 0, True), (None, 'script text', True, 0, False), (None, 'script text', False, 0, False), ('file.sh', None, True, 1, False), ('file.sh', None, False, 1, True), ('file.sh', 'script text', True, 0, False), ('file.sh', 'script text', False, 0, False)):
        try:
            if open_succeeds:
                open_mock = patch(BUILTINS + '.open', mock_open(read_data='script text'))
            else:
                open_mock = patch(BUILTINS + '.open', side_effect=Exception())
            with open_mock as mocked_open:
                try:
                    self.driver._change_vm_script('https://vm-vcloud/api/vApp/vapp-8c57a5b6-e61b-48ca-8a78-3b70ee65ef6d', vm_script=vm_script_file, vm_script_text=vm_script_text)
                    returned_early_res = True
                except CallException:
                    returned_early_res = False
                self.assertEqual(mocked_open.call_count, open_call_count)
                self.assertEqual(returned_early_res, returned_early)
        except AssertionError:
            assertion_error = True
            print_parameterized_failure([('vm_script_file', vm_script_file), ('vm_script_text', vm_script_text), ('open_succeeds', open_succeeds), ('open_call_count', open_call_count), ('returned_early', returned_early)])
    if assertion_error:
        self.fail(msg='Assertion error(s) encountered. Details above.')