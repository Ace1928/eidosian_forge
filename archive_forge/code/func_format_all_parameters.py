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
def format_all_parameters(params, param_files, template_file=None, template_url=None):
    parameters = {}
    parameters.update(format_parameters(params))
    parameters.update(format_parameter_file(param_files, template_file, template_url))
    return parameters