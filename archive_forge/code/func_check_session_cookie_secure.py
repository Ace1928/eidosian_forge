from django.conf import settings
from .. import Tags, Warning, register
@register(Tags.security, deploy=True)
def check_session_cookie_secure(app_configs, **kwargs):
    if settings.SESSION_COOKIE_SECURE is True:
        return []
    errors = []
    if _session_app():
        errors.append(W010)
    if _session_middleware():
        errors.append(W011)
    if len(errors) > 1:
        errors = [W012]
    return errors