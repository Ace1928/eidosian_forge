import argparse
from copy import deepcopy
import io
import json
import os
from unittest import mock
import sys
import tempfile
import testtools
from glanceclient.common import utils
from glanceclient import exc
from glanceclient import shell
from glanceclient.v2 import shell as test_shell  # noqa
def test_do_md_object_property_show_non_existing(self):
    args = self._make_args({'namespace': 'MyNamespace', 'object': 'MyObject', 'property': 'MyProperty', 'max_column_width': 80})
    with mock.patch.object(self.gc.metadefs_object, 'get') as mocked_get:
        expect_object = {'name': 'MyObject', 'properties': {}}
        mocked_get.return_value = expect_object
        self.assertRaises(SystemExit, test_shell.do_md_object_property_show, self.gc, args)
        mocked_get.assert_called_once_with('MyNamespace', 'MyObject')