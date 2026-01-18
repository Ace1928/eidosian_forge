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
def test_do_md_namespace_update(self):
    args = self._make_args({'id': 'MyNamespace', 'protected': True})
    with mock.patch.object(self.gc.metadefs_namespace, 'update') as mocked_update:
        expect_namespace = {'namespace': 'MyNamespace', 'protected': True}
        mocked_update.return_value = expect_namespace
        test_shell.do_md_namespace_update(self.gc, args)
        mocked_update.assert_called_once_with('MyNamespace', id='MyNamespace', protected=True)
        utils.print_dict.assert_called_once_with(expect_namespace)