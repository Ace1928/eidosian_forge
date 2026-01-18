from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from .. import Error, Tags, Warning, register
@register(Tags.security, deploy=True)
def check_allowed_hosts(app_configs, **kwargs):
    return [] if settings.ALLOWED_HOSTS else [W020]