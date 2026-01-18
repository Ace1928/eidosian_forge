from unittest import mock
from os_brick import exception
from os_brick.initiator import linuxrbd
from os_brick.tests import base
from os_brick import utils
def mock_read(offset, length):
    return self.full_data[offset:length]