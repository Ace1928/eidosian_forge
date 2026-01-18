import io
import logging
from unittest import mock
from oslotest import base as test_base
from oslo_log import rate_limit
def restore_handlers(logger, handlers):
    for handler in handlers:
        logger.addHandler(handler)