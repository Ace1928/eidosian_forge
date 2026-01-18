import io
import logging
from unittest import mock
from oslotest import base as test_base
from oslo_log import rate_limit
def install_filter(self, *args):
    rate_limit.install_filter(*args)
    logger = logging.getLogger()

    def restore_handlers(logger, handlers):
        for handler in handlers:
            logger.addHandler(handler)
    self.addCleanup(restore_handlers, logger, list(logger.handlers))
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    return (logger, stream)