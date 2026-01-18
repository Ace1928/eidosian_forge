import configparser
import logging
import logging.config
import logging.handlers
import os
import platform
import sys
from oslo_config import cfg
from oslo_utils import eventletutils
from oslo_utils import importutils
from oslo_utils import units
from oslo_log._i18n import _
from oslo_log import _options
from oslo_log import formatters
from oslo_log import handlers
def logging_excepthook(exc_type, value, tb):
    extra = {'exc_info': (exc_type, value, tb)}
    getLogger(product_name).critical('Unhandled error', **extra)