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
def apply_mapping_filter(identity_provider, protocol, assertion, resource_api, federation_api, identity_api):
    idp = federation_api.get_idp(identity_provider)
    utils.validate_idp(idp, protocol, assertion)
    mapped_properties, mapping_id = federation_api.evaluate(identity_provider, protocol, assertion)
    group_ids = mapped_properties['group_ids']
    utils.validate_mapped_group_ids(group_ids, mapping_id, identity_api)
    group_ids.extend(utils.transform_to_group_ids(mapped_properties['group_names'], mapping_id, identity_api, resource_api))
    mapped_properties['group_ids'] = list(set(group_ids))
    return (mapped_properties, mapping_id)