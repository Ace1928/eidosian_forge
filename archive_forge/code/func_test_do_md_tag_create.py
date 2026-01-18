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
def test_do_md_tag_create(self):
    args = self._make_args({'namespace': 'MyNamespace', 'name': 'MyTag'})
    with mock.patch.object(self.gc.metadefs_tag, 'create') as mocked_create:
        expect_tag = {'namespace': 'MyNamespace', 'name': 'MyTag'}
        mocked_create.return_value = expect_tag
        test_shell.do_md_tag_create(self.gc, args)
        mocked_create.assert_called_once_with('MyNamespace', 'MyTag')
        utils.print_dict.assert_called_once_with(expect_tag)