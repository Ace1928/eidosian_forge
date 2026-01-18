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
def test_do_md_object_create_invalid_schema(self):
    args = self._make_args({'namespace': 'MyNamespace', 'name': 'MyObject', 'schema': 'Invalid'})
    self.assertRaises(SystemExit, test_shell.do_md_object_create, self.gc, args)