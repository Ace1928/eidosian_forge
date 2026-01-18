from functools import update_wrapper, wraps
import logging; log = logging.getLogger(__name__)
import sys
import weakref
from warnings import warn
from passlib import exc, registry
from passlib.context import CryptContext
from passlib.exc import PasslibRuntimeWarning
from passlib.utils.compat import get_method_function, iteritems, OrderedDict, unicode
from passlib.utils.decor import memoized_property
def get_preset_config(name):
    """Returns configuration string for one of the preset strings
    supported by the ``PASSLIB_CONFIG`` setting.
    Currently supported presets:

    * ``"passlib-default"`` - default config used by this release of passlib.
    * ``"django-default"`` - config matching currently installed django version.
    * ``"django-latest"`` - config matching newest django version (currently same as ``"django-1.6"``).
    * ``"django-1.0"`` - config used by stock Django 1.0 - 1.3 installs
    * ``"django-1.4"`` - config used by stock Django 1.4 installs
    * ``"django-1.6"`` - config used by stock Django 1.6 installs
    """
    if name == 'django-default':
        if not DJANGO_VERSION:
            raise ValueError("can't resolve django-default preset, django not installed")
        name = 'django-1.6'
    if name == 'passlib-default':
        return PASSLIB_DEFAULT
    try:
        attr = _preset_map[name]
    except KeyError:
        raise ValueError('unknown preset config name: %r' % name)
    import passlib.apps
    return getattr(passlib.apps, attr).to_string()