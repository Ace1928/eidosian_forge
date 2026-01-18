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
def is_client_exception(self, ex):
    return isinstance(ex, exceptions.ClientException)