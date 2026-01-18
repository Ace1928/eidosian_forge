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
def list_container_networks(networks):
    columns = ('net_id', 'port_id', 'fixed_ips')
    utils.print_list(networks, columns, {'fixed_ips': format_network_fixed_ips}, sortby_index=None)