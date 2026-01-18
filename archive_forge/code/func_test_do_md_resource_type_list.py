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
def test_do_md_resource_type_list(self):
    args = self._make_args({})
    with mock.patch.object(self.gc.metadefs_resource_type, 'list') as mocked_list:
        expect_objects = ['MyResourceType1', 'MyResourceType2']
        mocked_list.return_value = expect_objects
        test_shell.do_md_resource_type_list(self.gc, args)
        self.assertEqual(1, mocked_list.call_count)