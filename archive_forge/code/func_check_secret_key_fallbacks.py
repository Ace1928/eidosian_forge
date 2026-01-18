from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from .. import Error, Tags, Warning, register
@register(Tags.security, deploy=True)
def check_secret_key_fallbacks(app_configs, **kwargs):
    warnings = []
    try:
        fallbacks = settings.SECRET_KEY_FALLBACKS
    except (ImproperlyConfigured, AttributeError):
        warnings.append(Warning(W025.msg % 'SECRET_KEY_FALLBACKS', id=W025.id))
    else:
        for index, key in enumerate(fallbacks):
            if not _check_secret_key(key):
                warnings.append(Warning(W025.msg % f'SECRET_KEY_FALLBACKS[{index}]', id=W025.id))
    return warnings