from django.conf import DEFAULT_STORAGE_ALIAS, STATICFILES_STORAGE_ALIAS, settings
from django.core.exceptions import ImproperlyConfigured
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
def create_storage(self, params):
    params = params.copy()
    backend = params.pop('BACKEND')
    options = params.pop('OPTIONS', {})
    try:
        storage_cls = import_string(backend)
    except ImportError as e:
        raise InvalidStorageError(f'Could not find backend {backend!r}: {e}') from e
    return storage_cls(**options)