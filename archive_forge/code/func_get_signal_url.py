import datetime
import email.utils
import hashlib
import logging
import random
import time
from urllib import parse
from oslo_config import cfg
from swiftclient import client as sc
from swiftclient import exceptions
from swiftclient import utils as swiftclient_utils
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_plugin
def get_signal_url(self, container_name, obj_name, timeout=None):
    """Turn on object versioning.

        We can use a single TempURL for multiple signals and return a Swift
        TempURL.
        """
    self.client().put_container(container_name, headers={'x-versions-location': container_name})
    self.client().put_object(container_name, obj_name, IN_PROGRESS)
    return self.get_temp_url(container_name, obj_name, timeout)