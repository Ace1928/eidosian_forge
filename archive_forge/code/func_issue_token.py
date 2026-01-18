import base64
import datetime
import uuid
from oslo_log import log
from oslo_utils import timeutils
from keystone.common import cache
from keystone.common import manager
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.federation import constants
from keystone.i18n import _
from keystone.models import token_model
from keystone import notifications
def issue_token(self, user_id, method_names, expires_at=None, system=None, project_id=None, domain_id=None, auth_context=None, trust_id=None, app_cred_id=None, thumbprint=None, parent_audit_id=None):
    token = token_model.TokenModel()
    token.methods = method_names
    token.system = system
    token.domain_id = domain_id
    token.project_id = project_id
    token.trust_id = trust_id
    token.application_credential_id = app_cred_id
    token.audit_id = random_urlsafe_str()
    token.oauth2_thumbprint = thumbprint
    token.parent_audit_id = parent_audit_id
    if auth_context:
        if constants.IDENTITY_PROVIDER in auth_context:
            token.is_federated = True
            token.protocol_id = auth_context[constants.PROTOCOL]
            idp_id = auth_context[constants.IDENTITY_PROVIDER]
            if isinstance(idp_id, bytes):
                idp_id = idp_id.decode('utf-8')
            token.identity_provider_id = idp_id
            token.user_id = auth_context['user_id']
            token.federated_groups = [{'id': group} for group in auth_context['group_ids']]
        if 'access_token_id' in auth_context:
            token.access_token_id = auth_context['access_token_id']
    if not token.user_id:
        token.user_id = user_id
    token.user_domain_id = token.user['domain_id']
    if isinstance(expires_at, datetime.datetime):
        token.expires_at = utils.isotime(expires_at, subsecond=True)
    if isinstance(expires_at, str):
        token.expires_at = expires_at
    elif not expires_at:
        token.expires_at = utils.isotime(default_expire_time(), subsecond=True)
    if app_cred_id is not None:
        app_cred_api = PROVIDERS.application_credential_api
        app_cred = app_cred_api.get_application_credential(token.application_credential_id)
        token_time = timeutils.normalize_time(timeutils.parse_isotime(token.expires_at))
        if app_cred['expires_at'] is not None and token_time > app_cred['expires_at']:
            token.expires_at = app_cred['expires_at'].isoformat()
            LOG.debug('Resetting token expiration to the application credential expiration: %s', app_cred['expires_at'].isoformat())
    token_id, issued_at = self.driver.generate_id_and_issued_at(token)
    token.mint(token_id, issued_at)
    if CONF.token.cache_on_issue or CONF.token.caching:
        self._validate_token.set(token, self, token.id)
    return token