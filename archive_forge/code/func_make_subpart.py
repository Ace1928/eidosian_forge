import collections
import email
from email.mime import multipart
from email.mime import text
import os
import pkgutil
import string
from urllib import parse as urlparse
from neutronclient.common import exceptions as q_exceptions
from novaclient import api_versions
from novaclient import client as nc
from novaclient import exceptions
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import netutils
import tenacity
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_exception
from heat.engine.clients import client_plugin
from heat.engine.clients import microversion_mixin
from heat.engine.clients import os as os_client
from heat.engine import constraints
def make_subpart(content, filename, subtype=None):
    if subtype is None:
        subtype = os.path.splitext(filename)[0]
    if content is None:
        content = ''
    try:
        content.encode('us-ascii')
        charset = 'us-ascii'
    except UnicodeEncodeError:
        charset = 'utf-8'
    msg = text.MIMEText(content, _subtype=subtype, _charset=charset) if subtype else text.MIMEText(content, _charset=charset)
    msg.add_header('Content-Disposition', 'attachment', filename=filename)
    return msg