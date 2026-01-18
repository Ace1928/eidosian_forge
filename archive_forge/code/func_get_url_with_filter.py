import contextlib
import os
import re
import textwrap
import time
from urllib import parse
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import uuidutils
import prettytable
from novaclient import exceptions
from novaclient.i18n import _
def get_url_with_filter(url, filters):
    query_string = prepare_query_string(filters)
    url = '%s%s' % (url, query_string)
    return url