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
def test_do_md_property_update_invalid_schema(self):
    args = self._make_args({'namespace': 'MyNamespace', 'property': 'MyProperty', 'name': 'MyObject', 'title': 'Title', 'schema': 'Invalid'})
    self.assertRaises(SystemExit, test_shell.do_md_property_update, self.gc, args)