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
def test_do_member_delete_with_few_arguments(self):
    args = self._make_args({'image_id': None, 'member_id': 'MEM-01'})
    msg = 'Unable to delete member. Specify image_id and member_id'
    self.assert_exits_with_msg(func=test_shell.do_member_delete, func_args=args, err_msg=msg)