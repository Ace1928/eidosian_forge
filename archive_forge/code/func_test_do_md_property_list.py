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
def test_do_md_property_list(self):
    args = self._make_args({'namespace': 'MyNamespace'})
    with mock.patch.object(self.gc.metadefs_property, 'list') as mocked_list:
        expect_objects = [{'namespace': 'MyNamespace', 'property': 'MyProperty', 'title': 'MyTitle'}]
        mocked_list.return_value = expect_objects
        test_shell.do_md_property_list(self.gc, args)
        mocked_list.assert_called_once_with('MyNamespace')
        utils.print_list.assert_called_once_with(expect_objects, ['name', 'title', 'type'])