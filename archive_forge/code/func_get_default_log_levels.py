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
def get_default_log_levels():
    """Return the Oslo Logging default log levels.

    Returns a copy of the list so an application can change the value
    and not affect the default value used in the log_opts configuration
    setup.
    """
    return list(_options.DEFAULT_LOG_LEVELS)