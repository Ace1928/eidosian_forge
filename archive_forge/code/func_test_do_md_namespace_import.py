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
def test_do_md_namespace_import(self):
    args = self._make_args({'file': 'test'})
    expect_namespace = {'namespace': 'MyNamespace', 'protected': True}
    with mock.patch.object(self.gc.metadefs_namespace, 'create') as mocked_create:
        mock_read = mock.Mock(return_value=json.dumps(expect_namespace))
        mock_file = mock.Mock(read=mock_read)
        utils.get_data_file = mock.Mock(return_value=mock_file)
        mocked_create.return_value = expect_namespace
        test_shell.do_md_namespace_import(self.gc, args)
        mocked_create.assert_called_once_with(**expect_namespace)
        utils.print_dict.assert_called_once_with(expect_namespace)