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
def parse_health(hc_str):
    err_msg = "Invalid healthcheck argument '%s'. healthcheck arguments must be of the form --healthcheck <cmd='command',interval=time,retries=integer,timeout=time>, and the unit of time is s(seconds), m(minutes), h(hours)." % hc_str
    keys = ['cmd', 'interval', 'retries', 'timeout']
    health_info = {}
    for kv_str in hc_str[0].split(','):
        try:
            k, v = kv_str.split('=', 1)
            k = k.strip()
            v = v.strip()
        except ValueError:
            raise apiexec.CommandError(err_msg)
        if k in keys:
            if health_info.get(k):
                raise apiexec.CommandError(err_msg)
            elif k in ['interval', 'timeout']:
                health_info[k] = _convert_healthcheck_para(v, err_msg)
            elif k == 'retries':
                health_info[k] = int(v)
            else:
                health_info[k] = v
        else:
            raise apiexec.CommandError(err_msg)
    return health_info