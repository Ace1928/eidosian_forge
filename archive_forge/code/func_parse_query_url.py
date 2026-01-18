import base64
import logging
import os
import textwrap
import uuid
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
import prettytable
from urllib import error
from urllib import parse
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient import exc
def parse_query_url(url):
    base_url, query_params = url.split('?')
    return (base_url, parse.parse_qs(query_params))