import flask
from oslo_log import log
from keystone.auth import core
from keystone.common import provider_api
from keystone import exception
from keystone.federation import constants
from keystone.i18n import _
from keystone.receipt import handlers as receipt_handlers
def authenticate_for_token(auth=None):
    """Authenticate user and issue a token."""
    try:
        auth_info = core.AuthInfo.create(auth=auth)
        auth_context = core.AuthContext(method_names=[], bind={})
        authenticate(auth_info, auth_context)
        if auth_context.get('access_token_id'):
            auth_info.set_scope(None, auth_context['project_id'], None)
        _check_and_set_default_scoping(auth_info, auth_context)
        domain_id, project_id, trust, unscoped, system = auth_info.get_scope()
        trust_id = trust.get('id') if trust else None
        receipt = receipt_handlers.extract_receipt(auth_context)
        if receipt:
            method_names_set = set(auth_context.get('method_names', []) + receipt.methods)
        else:
            method_names_set = set(auth_context.get('method_names', []))
        method_names = list(method_names_set)
        app_cred_id = None
        if 'application_credential' in method_names:
            token_auth = auth_info.auth['identity']
            app_cred_id = token_auth['application_credential']['id']
        if not core.UserMFARulesValidator.check_auth_methods_against_rules(auth_context['user_id'], method_names_set):
            raise exception.InsufficientAuthMethods(user_id=auth_context['user_id'], methods=method_names)
        expires_at = auth_context.get('expires_at')
        token_audit_id = auth_context.get('audit_id')
        token = PROVIDERS.token_provider_api.issue_token(auth_context['user_id'], method_names, expires_at=expires_at, system=system, project_id=project_id, domain_id=domain_id, auth_context=auth_context, trust_id=trust_id, app_cred_id=app_cred_id, parent_audit_id=token_audit_id)
        if trust:
            PROVIDERS.trust_api.consume_use(token.trust_id)
        return token
    except exception.TrustNotFound as e:
        LOG.warning(e)
        raise exception.Unauthorized(e)