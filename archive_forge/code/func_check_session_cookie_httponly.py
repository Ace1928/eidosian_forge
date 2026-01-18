from django.conf import settings
from .. import Tags, Warning, register
@register(Tags.security, deploy=True)
def check_session_cookie_httponly(app_configs, **kwargs):
    if settings.SESSION_COOKIE_HTTPONLY is True:
        return []
    errors = []
    if _session_app():
        errors.append(W013)
    if _session_middleware():
        errors.append(W014)
    if len(errors) > 1:
        errors = [W015]
    return errors