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
def test_do_md_namespace_show(self):
    args = self._make_args({'namespace': 'MyNamespace', 'max_column_width': 80, 'resource_type': None})
    with mock.patch.object(self.gc.metadefs_namespace, 'get') as mocked_get:
        expect_namespace = {'namespace': 'MyNamespace'}
        mocked_get.return_value = expect_namespace
        test_shell.do_md_namespace_show(self.gc, args)
        mocked_get.assert_called_once_with('MyNamespace')
        utils.print_dict.assert_called_once_with(expect_namespace, 80)