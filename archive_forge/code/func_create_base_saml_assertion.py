from oslo_serialization import jsonutils
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.federation import idp as keystone_idp
from keystone.federation import utils as federation_utils
from keystone.i18n import _
def create_base_saml_assertion(auth):
    issuer = CONF.saml.idp_entity_id
    sp_id = auth['scope']['service_provider']['id']
    service_provider = PROVIDERS.federation_api.get_sp(sp_id)
    federation_utils.assert_enabled_service_provider_object(service_provider)
    sp_url = service_provider['sp_url']
    token_id = auth['identity']['token']['id']
    token = PROVIDERS.token_provider_api.validate_token(token_id)
    if not token.project_scoped:
        action = _('Use a project scoped token when attempting to create a SAML assertion')
        raise exception.ForbiddenAction(action=action)
    subject = token.user['name']
    role_names = []
    for role in token.roles:
        role_names.append(role['name'])
    project = token.project['name']
    project_domain_name = token.project_domain['name']
    subject_domain_name = token.user_domain['name']

    def group_membership():
        """Return a list of dictionaries serialized as strings.

        The expected return structure is::

        ['JSON:{"name":"group1","domain":{"name":"Default"}}',
        'JSON:{"name":"group2","domain":{"name":"Default"}}']
        """
        user_groups = []
        groups = PROVIDERS.identity_api.list_groups_for_user(token.user_id)
        for group in groups:
            user_group = {}
            group_domain_name = PROVIDERS.resource_api.get_domain(group['domain_id'])['name']
            user_group['name'] = group['name']
            user_group['domain'] = {'name': group_domain_name}
            user_groups.append('JSON:' + jsonutils.dumps(user_group))
        return user_groups
    groups = group_membership()
    generator = keystone_idp.SAMLGenerator()
    response = generator.samlize_token(issuer, sp_url, subject, subject_domain_name, role_names, project, project_domain_name, groups)
    return (response, service_provider)