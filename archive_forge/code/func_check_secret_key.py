from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from .. import Error, Tags, Warning, register
@register(Tags.security, deploy=True)
def check_secret_key(app_configs, **kwargs):
    try:
        secret_key = settings.SECRET_KEY
    except (ImproperlyConfigured, AttributeError):
        passed_check = False
    else:
        passed_check = _check_secret_key(secret_key)
    return [] if passed_check else [W009]