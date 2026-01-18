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
def handle_unscoped_token(auth_payload, resource_api, federation_api, identity_api, assignment_api, role_api):

    def validate_shadow_mapping(shadow_projects, existing_roles, user_domain_id, idp_id):
        for shadow_project in shadow_projects:
            for shadow_role in shadow_project['roles']:
                if shadow_role['name'] not in existing_roles:
                    LOG.error('Role %s was specified in the mapping but does not exist. All roles specified in a mapping must exist before assignment.', shadow_role['name'])
                    raise exception.RoleNotFound(shadow_role['name'])
                role = existing_roles[shadow_role['name']]
                if role['domain_id'] is not None and role['domain_id'] != user_domain_id:
                    LOG.error('Role %(role)s is a domain-specific role and cannot be assigned within %(domain)s.', {'role': shadow_role['name'], 'domain': user_domain_id})
                    raise exception.DomainSpecificRoleNotWithinIdPDomain(role_name=shadow_role['name'], identity_provider=idp_id)

    def is_ephemeral_user(mapped_properties):
        return mapped_properties['user']['type'] == utils.UserType.EPHEMERAL

    def build_ephemeral_user_context(user, mapped_properties, identity_provider, protocol):
        resp = {}
        resp['user_id'] = user['id']
        resp['group_ids'] = mapped_properties['group_ids']
        resp[federation_constants.IDENTITY_PROVIDER] = identity_provider
        resp[federation_constants.PROTOCOL] = protocol
        return resp

    def build_local_user_context(mapped_properties):
        resp = {}
        user_info = auth_plugins.UserAuthInfo.create(mapped_properties, METHOD_NAME)
        resp['user_id'] = user_info.user_id
        return resp
    assertion = extract_assertion_data()
    try:
        identity_provider = auth_payload['identity_provider']
    except KeyError:
        raise exception.ValidationError(attribute='identity_provider', target='mapped')
    try:
        protocol = auth_payload['protocol']
    except KeyError:
        raise exception.ValidationError(attribute='protocol', target='mapped')
    utils.assert_enabled_identity_provider(federation_api, identity_provider)
    group_ids = None
    token_id = None
    user_id = None
    try:
        try:
            mapped_properties, mapping_id = apply_mapping_filter(identity_provider, protocol, assertion, resource_api, federation_api, identity_api)
        except exception.ValidationError as e:
            raise exception.Unauthorized(e)
        if is_ephemeral_user(mapped_properties):
            idp_domain_id = federation_api.get_idp(identity_provider)['domain_id']
            validate_and_prepare_federated_user(mapped_properties, idp_domain_id, resource_api)
            user = identity_api.shadow_federated_user(identity_provider, protocol, mapped_properties['user'], group_ids=mapped_properties['group_ids'])
            if 'projects' in mapped_properties:
                existing_roles = {role['name']: role for role in role_api.list_roles()}
                validate_shadow_mapping(mapped_properties['projects'], existing_roles, mapped_properties['user']['domain']['id'], identity_provider)
                handle_projects_from_mapping(mapped_properties['projects'], idp_domain_id, existing_roles, user, assignment_api, resource_api)
            user_id = user['id']
            group_ids = mapped_properties['group_ids']
            response_data = build_ephemeral_user_context(user, mapped_properties, identity_provider, protocol)
        else:
            response_data = build_local_user_context(mapped_properties)
    except Exception:
        outcome = taxonomy.OUTCOME_FAILURE
        notifications.send_saml_audit_notification('authenticate', user_id, group_ids, identity_provider, protocol, token_id, outcome)
        raise
    else:
        outcome = taxonomy.OUTCOME_SUCCESS
        notifications.send_saml_audit_notification('authenticate', user_id, group_ids, identity_provider, protocol, token_id, outcome)
    return response_data