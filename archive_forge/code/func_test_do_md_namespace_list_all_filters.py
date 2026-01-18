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
def test_do_md_namespace_list_all_filters(self):
    args = self._make_args({'resource_types': ['OS::Compute::Aggregate'], 'visibility': 'public', 'page_size': None})
    with mock.patch.object(self.gc.metadefs_namespace, 'list') as mocked_list:
        expect_namespaces = [{'namespace': 'MyNamespace'}]
        mocked_list.return_value = expect_namespaces
        test_shell.do_md_namespace_list(self.gc, args)
        mocked_list.assert_called_once_with(filters={'resource_types': ['OS::Compute::Aggregate'], 'visibility': 'public'})
        utils.print_list.assert_called_once_with(expect_namespaces, ['namespace'])