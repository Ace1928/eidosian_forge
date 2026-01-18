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
def link_formatter(links):

    def format_link(link):
        if 'rel' in link:
            return '%s (%s)' % (link.get('href', ''), link.get('rel', ''))
        else:
            return '%s' % link.get('href', '')
    return '\n'.join((format_link(link) for link in links or []))