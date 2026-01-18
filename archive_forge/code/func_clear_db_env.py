import copy
import functools
import uuid
from oslo_config import cfg
from oslo_log import log as logging
from glance.common import exception
from glance.common import timeutils
from glance.common import utils
from glance.db import utils as db_utils
from glance.i18n import _, _LI, _LW
def clear_db_env(*args, **kwargs):
    """
    Setup global environment configuration variables.

    We have no connection-oriented environment variables, so this is a NOOP.
    """
    pass