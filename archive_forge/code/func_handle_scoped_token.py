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
def handle_scoped_token(token, federation_api, identity_api):
    response_data = {}
    utils.validate_expiration(token)
    token_audit_id = token.audit_id
    identity_provider = token.identity_provider_id
    protocol = token.protocol_id
    user_id = token.user_id
    group_ids = []
    for group_dict in token.federated_groups:
        group_ids.append(group_dict['id'])
    send_notification = functools.partial(notifications.send_saml_audit_notification, 'authenticate', user_id, group_ids, identity_provider, protocol, token_audit_id)
    utils.assert_enabled_identity_provider(federation_api, identity_provider)
    try:
        mapping = federation_api.get_mapping_from_idp_and_protocol(identity_provider, protocol)
        utils.validate_mapped_group_ids(group_ids, mapping['id'], identity_api)
    except Exception:
        send_notification(taxonomy.OUTCOME_FAILURE)
        raise
    else:
        send_notification(taxonomy.OUTCOME_SUCCESS)
    response_data['user_id'] = user_id
    response_data['group_ids'] = group_ids
    response_data[federation_constants.IDENTITY_PROVIDER] = identity_provider
    response_data[federation_constants.PROTOCOL] = protocol
    return response_data