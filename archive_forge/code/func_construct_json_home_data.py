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
def construct_json_home_data(rel, status=json_home.Status.STABLE, path_vars=None, resource_relation_func=_v3_resource_relation):
    rel = resource_relation_func(resource_name=rel)
    return JsonHomeData(rel=rel, status=status, path_vars=path_vars or {})