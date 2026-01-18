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
def is_valid_temp_url_path(self, path):
    """Return True if path is a valid Swift TempURL path, False otherwise.

        A Swift TempURL path must:
        - Be five parts, ['', 'v1', 'account', 'container', 'object']
        - Be a v1 request
        - Have account, container, and object values
        - Have an object value with more than just '/'s

        :param path: The TempURL path
        :type path: string
        """
    parts = path.split('/', 4)
    return bool(len(parts) == 5 and (not parts[0]) and (parts[1] == 'v1') and parts[2].endswith(self.context.tenant_id) and parts[3] and parts[4].strip('/'))