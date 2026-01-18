import copy
from collections import defaultdict
from django.conf import settings
from django.template.backends.django import get_template_tag_modules
from . import Error, Tags, Warning, register
@register(Tags.templates)
def check_setting_app_dirs_loaders(app_configs, **kwargs):
    return [E001] if any((conf.get('APP_DIRS') and 'loaders' in conf.get('OPTIONS', {}) for conf in settings.TEMPLATES)) else []