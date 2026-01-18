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
def setup_paste_factories(conf):
    """Set up the generic paste app and filter factories.

    Set things up so that:

      paste.app_factory = heat.common.wsgi:app_factory

    and

      paste.filter_factory = heat.common.wsgi:filter_factory

    work correctly while loading PasteDeploy configuration.

    The app factories are constructed at runtime to allow us to pass a
    ConfigOpts object to the WSGI classes.

    :param conf: a ConfigOpts object
    """
    global app_factory, filter_factory
    app_factory = AppFactory(conf)
    filter_factory = FilterFactory(conf)