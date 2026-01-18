import inspect
from django.conf import settings
from .. import Error, Tags, Warning, register
@register(Tags.security, deploy=True)
def check_csrf_middleware(app_configs, **kwargs):
    passed_check = _csrf_middleware()
    return [] if passed_check else [W003]