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
def format_container_addresses(container):
    addresses = getattr(container, 'addresses', {})
    output = []
    networks = []
    try:
        for address_name, address_list in addresses.items():
            for a in address_list:
                output.append(a['addr'])
            networks.append(address_name)
    except Exception:
        pass
    setattr(container, 'addresses', ', '.join(output))
    setattr(container, 'networks', ', '.join(networks))
    container._info['addresses'] = ', '.join(output)
    container._info['networks'] = ', '.join(networks)