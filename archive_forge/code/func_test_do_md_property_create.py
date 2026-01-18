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
def test_do_md_property_create(self):
    args = self._make_args({'namespace': 'MyNamespace', 'name': 'MyProperty', 'title': 'Title', 'type': 'boolean', 'schema': '{}'})
    with mock.patch.object(self.gc.metadefs_property, 'create') as mocked_create:
        expect_property = {'namespace': 'MyNamespace', 'name': 'MyProperty', 'title': 'Title', 'type': 'boolean'}
        mocked_create.return_value = expect_property
        test_shell.do_md_property_create(self.gc, args)
        mocked_create.assert_called_once_with('MyNamespace', name='MyProperty', title='Title', type='boolean')
        utils.print_dict.assert_called_once_with(expect_property)