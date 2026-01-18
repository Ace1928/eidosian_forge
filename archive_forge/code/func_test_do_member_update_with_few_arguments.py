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
def test_do_member_update_with_few_arguments(self):
    input = {'image_id': 'IMG-01', 'member_id': 'MEM-01', 'member_status': None}
    args = self._make_args(input)
    msg = 'Unable to update member. Specify image_id, member_id and member_status'
    self.assert_exits_with_msg(func=test_shell.do_member_update, func_args=args, err_msg=msg)