import abc
import collections
import functools
import re
import uuid
import wsgiref.util
import flask
from flask import blueprints
import flask_restful
import flask_restful.utils
import http.client
from oslo_log import log
from oslo_log import versionutils
from oslo_serialization import jsonutils
from keystone.common import authorization
from keystone.common import context
from keystone.common import driver_hints
from keystone.common import json_home
from keystone.common.rbac_enforcer import enforcer
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
@classmethod
def get_token_ref(cls):
    """Retrieve KeystoneToken object from the auth context and returns it.

        :raises keystone.exception.Unauthorized: If auth context cannot be
                                                 found.
        :returns: The KeystoneToken object.
        """
    try:
        auth_context = flask.request.environ.get(authorization.AUTH_CONTEXT_ENV, {})
        return auth_context['token']
    except KeyError:
        LOG.warning("Couldn't find the auth context.")
        raise exception.Unauthorized()