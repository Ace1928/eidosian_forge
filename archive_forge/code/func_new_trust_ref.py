import atexit
import base64
import contextlib
import datetime
import functools
import hashlib
import json
import secrets
import ldap
import os
import shutil
import socket
import sys
import uuid
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography import x509
import fixtures
import flask
from flask import testing as flask_testing
import http.client
from oslo_config import fixture as config_fixture
from oslo_context import context as oslo_context
from oslo_context import fixture as oslo_ctx_fixture
from oslo_log import fixture as log_fixture
from oslo_log import log
from oslo_utils import timeutils
import testtools
from testtools import testcase
import keystone.api
from keystone.common import context
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import sql
import keystone.conf
from keystone import exception
from keystone.identity.backends.ldap import common as ks_ldap
from keystone import notifications
from keystone.resource.backends import base as resource_base
from keystone.server.flask import application as flask_app
from keystone.server.flask import core as keystone_flask
from keystone.tests.unit import ksfixtures
def new_trust_ref(trustor_user_id, trustee_user_id, project_id=None, impersonation=None, expires=None, role_ids=None, role_names=None, remaining_uses=None, allow_redelegation=False, redelegation_count=None, **kwargs):
    ref = {'id': uuid.uuid4().hex, 'trustor_user_id': trustor_user_id, 'trustee_user_id': trustee_user_id, 'impersonation': impersonation or False, 'project_id': project_id, 'remaining_uses': remaining_uses, 'allow_redelegation': allow_redelegation}
    if isinstance(redelegation_count, int):
        ref.update(redelegation_count=redelegation_count)
    if isinstance(expires, str):
        ref['expires_at'] = expires
    elif isinstance(expires, dict):
        ref['expires_at'] = (timeutils.utcnow() + datetime.timedelta(**expires)).strftime(TIME_FORMAT)
    elif expires is None:
        pass
    else:
        raise NotImplementedError('Unexpected value for "expires"')
    role_ids = role_ids or []
    role_names = role_names or []
    if role_ids or role_names:
        ref['roles'] = []
        for role_id in role_ids:
            ref['roles'].append({'id': role_id})
        for role_name in role_names:
            ref['roles'].append({'name': role_name})
    ref.update(kwargs)
    return ref