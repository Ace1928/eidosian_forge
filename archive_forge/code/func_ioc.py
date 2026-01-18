import json
import os
import urllib
from oslo_log import log as logging
import requests
from os_brick import exception
from os_brick.i18n import _
from os_brick import initiator
from os_brick.initiator.connectors import base
from os_brick.privileged import scaleio as priv_scaleio
from os_brick import utils
def ioc(direction, _type, nr, size):
    """Implementation of _IOC macro from <sys/ioctl.h>."""
    return direction | (size & 8191) << 16 | ord(_type) << 8 | nr