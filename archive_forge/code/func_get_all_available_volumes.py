import warnings
from os_win import utilsfactory
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick import initiator
from os_brick.initiator import initiator_connector
from os_brick import utils
def get_all_available_volumes(self):
    return []