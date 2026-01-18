import builtins
import functools
import io
import time
from unittest import mock
import ddt
from os_brick import exception
from os_brick.tests import base
from os_brick import utils
@utils.retry(exception.VolumeDeviceNotFound, interval, retries, backoff_rate)
def always_fails():
    self.counter += 1
    raise exception.VolumeDeviceNotFound(device='fake')