import logging
import os
import sys
import warnings
from logging.handlers import WatchedFileHandler
from kombu.utils.encoding import set_default_encoding_file
from celery import signals
from celery._state import get_current_task
from celery.exceptions import CDeprecationWarning, CPendingDeprecationWarning
from celery.local import class_property
from celery.utils.log import (ColorFormatter, LoggingProxy, get_logger, get_multiprocessing_logger, mlevel,
from celery.utils.nodenames import node_format
from celery.utils.term import colored
def setup_handlers(self, logger, logfile, format, colorize, formatter=ColorFormatter, **kwargs):
    if self._is_configured(logger):
        return logger
    handler = self._detect_handler(logfile)
    handler.setFormatter(formatter(format, use_color=colorize))
    logger.addHandler(handler)
    return logger