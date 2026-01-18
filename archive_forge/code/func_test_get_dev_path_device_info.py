import builtins
import functools
import io
import time
from unittest import mock
import ddt
from os_brick import exception
from os_brick.tests import base
from os_brick import utils
@ddt.data({'con_props': {}, 'dev_info': {'path': '/dev/sda'}}, {'con_props': {}, 'dev_info': {'path': b'/dev/sda'}}, {'con_props': None, 'dev_info': {'path': '/dev/sda'}}, {'con_props': None, 'dev_info': {'path': b'/dev/sda'}}, {'con_props': {'device_path': b'/dev/sdb'}, 'dev_info': {'path': '/dev/sda'}}, {'con_props': {'device_path': '/dev/sdb'}, 'dev_info': {'path': b'/dev/sda'}})
@ddt.unpack
def test_get_dev_path_device_info(self, con_props, dev_info):
    self.assertEqual('/dev/sda', utils.get_dev_path(con_props, dev_info))