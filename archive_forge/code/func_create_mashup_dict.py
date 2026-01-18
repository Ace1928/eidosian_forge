import errno
from eventlet.green import socket
import functools
import os
import re
import urllib
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import netutils
from oslo_utils import strutils
from webob import exc
from glance.common import exception
from glance.common import location_strategy
from glance.common import timeutils
from glance.common import wsgi
from glance.i18n import _, _LE, _LW
def create_mashup_dict(image_meta):
    """
    Returns a dictionary-like mashup of the image core properties
    and the image custom properties from given image metadata.

    :param image_meta: metadata of image with core and custom properties
    """
    d = {}
    for key, value in image_meta.items():
        if isinstance(value, dict):
            for subkey, subvalue in create_mashup_dict(value).items():
                if subkey not in image_meta:
                    d[subkey] = subvalue
        else:
            d[key] = value
    return d