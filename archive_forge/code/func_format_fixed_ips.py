import base64
import binascii
import os
import re
import shlex
from oslo_serialization import jsonutils
from oslo_utils import netutils
from urllib import parse
from urllib import request
from zunclient.common.apiclient import exceptions as apiexec
from zunclient.common import cliutils as utils
from zunclient import exceptions as exc
from zunclient.i18n import _
def format_fixed_ips(fixed_ips):
    if fixed_ips is None:
        return None
    return ','.join([fip['ip_address'] for fip in fixed_ips])