import functools
import gettext as gettext_module
import os
import re
import sys
import warnings
from asgiref.local import Local
from django.apps import apps
from django.conf import settings
from django.conf.locale import LANG_INFO
from django.core.exceptions import AppRegistryNotReady
from django.core.signals import setting_changed
from django.dispatch import receiver
from django.utils.regex_helper import _lazy_re_compile
from django.utils.safestring import SafeData, mark_safe
from . import to_language, to_locale
def all_locale_paths():
    """
    Return a list of paths to user-provides languages files.
    """
    globalpath = os.path.join(os.path.dirname(sys.modules[settings.__module__].__file__), 'locale')
    app_paths = []
    for app_config in apps.get_app_configs():
        locale_path = os.path.join(app_config.path, 'locale')
        if os.path.exists(locale_path):
            app_paths.append(locale_path)
    return [globalpath, *settings.LOCALE_PATHS, *app_paths]