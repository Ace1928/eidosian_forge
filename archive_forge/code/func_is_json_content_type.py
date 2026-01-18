import abc
import errno
import os
import signal
import sys
import time
import eventlet
from eventlet.green import socket
from eventlet.green import ssl
import eventlet.greenio
import eventlet.wsgi
import functools
from oslo_concurrency import processutils
from oslo_config import cfg
import oslo_i18n as i18n
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import importutils
from paste.deploy import loadwsgi
from routes import middleware
import webob.dec
import webob.exc
from heat.api.aws import exception as aws_exception
from heat.common import exception
from heat.common.i18n import _
from heat.common import serializers
def is_json_content_type(request):
    if request.method == 'GET':
        try:
            aws_content_type = request.params.get('ContentType')
        except Exception:
            aws_content_type = None
        content_type = aws_content_type or request.content_type
    else:
        content_type = request.content_type
    if not content_type or content_type.startswith('text/plain'):
        content_type = 'application/json'
    if content_type in ('JSON', 'application/json') and request.body.startswith(b'{'):
        return True
    return False