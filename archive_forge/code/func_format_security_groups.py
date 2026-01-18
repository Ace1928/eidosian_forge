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
def format_security_groups(groups):
    return ', '.join((group['name'] for group in groups))