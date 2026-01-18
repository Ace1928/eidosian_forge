from unittest import mock
from oslotest import base as test_base
from oslo_service import fixture
from oslo_service import loopingcall
@loopingcall.RetryDecorator(max_retry_count=3, inc_sleep_time=2, exceptions=(ValueError,))
def retried_method():
    raise ValueError('!')