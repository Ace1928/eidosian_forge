import functools
import logging
import re
from oslo_utils import strutils
from cinderclient._i18n import _
from cinderclient import exceptions
from cinderclient import utils
def get_highest_version(client):
    """Queries the server version info and returns highest supported
    microversion

    :param client: client object
    :returns: APIVersion
    """
    server_start_version, server_end_version = _get_server_version_range(client)
    return server_end_version