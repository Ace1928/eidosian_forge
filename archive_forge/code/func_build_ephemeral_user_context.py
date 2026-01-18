import functools
import uuid
import flask
from oslo_log import log
from pycadf import cadftaxonomy as taxonomy
from urllib import parse
from keystone.auth import plugins as auth_plugins
from keystone.auth.plugins import base
from keystone.common import provider_api
from keystone import exception
from keystone.federation import constants as federation_constants
from keystone.federation import utils
from keystone.i18n import _
from keystone import notifications
def build_ephemeral_user_context(user, mapped_properties, identity_provider, protocol):
    resp = {}
    resp['user_id'] = user['id']
    resp['group_ids'] = mapped_properties['group_ids']
    resp[federation_constants.IDENTITY_PROVIDER] = identity_provider
    resp[federation_constants.PROTOCOL] = protocol
    return resp