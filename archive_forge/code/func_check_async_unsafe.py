import os
from . import Error, Tags, register
@register(Tags.async_support, deploy=True)
def check_async_unsafe(app_configs, **kwargs):
    if os.environ.get('DJANGO_ALLOW_ASYNC_UNSAFE'):
        return [E001]
    return []