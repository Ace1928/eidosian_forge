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
def assert_exits_with_msg(self, func, func_args, err_msg=None):
    with mock.patch.object(utils, 'exit') as mocked_utils_exit:
        mocked_utils_exit.return_value = '%s' % err_msg
        func(self.gc, func_args)
        if err_msg:
            mocked_utils_exit.assert_called_once_with(err_msg)
        else:
            mocked_utils_exit.assert_called_once_with()