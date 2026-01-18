from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from .. import Error, Tags, Warning, register
@register(Tags.security, deploy=True)
def check_xframe_deny(app_configs, **kwargs):
    passed_check = not _xframe_middleware() or settings.X_FRAME_OPTIONS == 'DENY'
    return [] if passed_check else [W019]