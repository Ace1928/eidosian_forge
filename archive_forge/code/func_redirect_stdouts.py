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
def redirect_stdouts(self, loglevel=None, name='celery.redirected'):
    self.redirect_stdouts_to_logger(get_logger(name), loglevel=loglevel)
    os.environ.update(CELERY_LOG_REDIRECT='1', CELERY_LOG_REDIRECT_LEVEL=str(loglevel or ''))