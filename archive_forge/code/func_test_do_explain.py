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
def test_do_explain(self):
    input = {'page_size': 18, 'id': 'pass', 'schemas': 'test', 'model': 'test'}
    args = self._make_args(input)
    with mock.patch.object(utils, 'print_list'):
        test_shell.do_explain(self.gc, args)
        self.gc.schemas.get.assert_called_once_with('test')