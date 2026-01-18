import builtins
import functools
import io
import time
from unittest import mock
import ddt
from os_brick import exception
from os_brick.tests import base
from os_brick import utils
@utils.retry(retry=utils.retry_if_exit_code, retry_param=exit_code)
def raise_retriable_exit_code():
    raise exception(exit_code=exit_code)