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
@staticmethod
def build_driver_hints(supported_filters):
    """Build list hints based on the context query string.

        :param supported_filters: list of filters supported, so ignore any
                                  keys in query_dict that are not in this list.

        """
    hints = driver_hints.Hints()
    if not flask.request.args:
        return hints
    for key, value in flask.request.args.items(multi=True):
        if supported_filters is None or key in supported_filters:
            hints.add_filter(key, value)
            continue
        for valid_key in supported_filters:
            if not key.startswith(valid_key + '__'):
                continue
            base_key, comparator = key.split('__', 1)
            case_sensitive = True
            if comparator.startswith('i'):
                case_sensitive = False
                comparator = comparator[1:]
            hints.add_filter(base_key, value, comparator=comparator, case_sensitive=case_sensitive)
    return hints