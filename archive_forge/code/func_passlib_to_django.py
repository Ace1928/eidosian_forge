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
def passlib_to_django(self, passlib_hasher, cached=True):
    """
        Convert passlib hasher / name to Django hasher.

        :param passlib_hasher:
            passlib hasher / name

        :returns:
            django hasher instance
        """
    if not hasattr(passlib_hasher, 'name'):
        passlib_hasher = self._get_passlib_hasher(passlib_hasher)
    if cached:
        cache = self._django_hasher_cache
        try:
            return cache[passlib_hasher]
        except KeyError:
            pass
        result = cache[passlib_hasher] = self.passlib_to_django(passlib_hasher, cached=False)
        return result
    django_name = getattr(passlib_hasher, 'django_name', None)
    if django_name:
        return self._create_django_hasher(django_name)
    else:
        return _PasslibHasherWrapper(passlib_hasher)