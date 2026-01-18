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
def get_template_url(template_file=None, template_url=None):
    if template_file:
        template_url = normalise_file_path_to_url(template_file)
    return template_url