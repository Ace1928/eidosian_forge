from oslo_log import log
from oslo_serialization import msgpackutils
from oslo_utils import timeutils
from keystone.common import cache
from keystone.common import utils
def build_token_values(token):
    token_expires_at = timeutils.parse_isotime(token.expires_at)
    token_expires_at = token_expires_at.replace(microsecond=0)
    token_values = {'expires_at': timeutils.normalize_time(token_expires_at), 'issued_at': timeutils.normalize_time(timeutils.parse_isotime(token.issued_at)), 'audit_id': token.audit_id, 'audit_chain_id': token.parent_audit_id}
    if token.user_id is not None:
        token_values['user_id'] = token.user_id
        token_values['identity_domain_id'] = token.user_domain['id']
    else:
        token_values['user_id'] = None
        token_values['identity_domain_id'] = None
    if token.project_id is not None:
        token_values['project_id'] = token.project_id
        token_values['assignment_domain_id'] = token.project_domain['id']
    else:
        token_values['project_id'] = None
    if token.domain_id is not None:
        token_values['assignment_domain_id'] = token.domain_id
    else:
        token_values['assignment_domain_id'] = None
    role_list = []
    token_roles = token.roles
    if token_roles is not None:
        for role in token_roles:
            role_list.append(role['id'])
    token_values['roles'] = role_list
    if token.trust_scoped:
        token_values['trust_id'] = token.trust['id']
        token_values['trustor_id'] = token.trustor['id']
        token_values['trustee_id'] = token.trustee['id']
    else:
        token_values['trust_id'] = None
        token_values['trustor_id'] = None
        token_values['trustee_id'] = None
    if token.oauth_scoped:
        token_values['consumer_id'] = token.access_token['consumer_id']
        token_values['access_token_id'] = token.access_token['id']
    else:
        token_values['consumer_id'] = None
        token_values['access_token_id'] = None
    return token_values