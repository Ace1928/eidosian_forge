from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from .. import Error, Tags, Warning, register
@register(Tags.security, deploy=True)
def check_sts_preload(app_configs, **kwargs):
    passed_check = not _security_middleware() or not settings.SECURE_HSTS_SECONDS or settings.SECURE_HSTS_PRELOAD is True
    return [] if passed_check else [W021]