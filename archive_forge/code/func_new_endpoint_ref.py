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
def new_endpoint_ref(service_id, interface='public', region_id=NEEDS_REGION_ID, **kwargs):
    ref = {'id': uuid.uuid4().hex, 'name': uuid.uuid4().hex, 'description': uuid.uuid4().hex, 'interface': interface, 'service_id': service_id, 'url': 'https://' + uuid.uuid4().hex + '.com'}
    if region_id is NEEDS_REGION_ID:
        ref['region_id'] = uuid.uuid4().hex
    elif region_id is None and kwargs.get('region') is not None:
        raise NotImplementedError('use new_endpoint_ref_with_region')
    else:
        ref['region_id'] = region_id
    ref.update(kwargs)
    return ref