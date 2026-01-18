from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_extension_utils
from _pydevd_bundle import pydevd_resolver
import sys
from _pydevd_bundle.pydevd_constants import BUILTINS_MODULE_NAME, MAXIMUM_VARIABLE_REPRESENTATION_SIZE, \
from _pydev_bundle.pydev_imports import quote
from _pydevd_bundle.pydevd_extension_api import TypeResolveProvider, StrPresentationProvider
from _pydevd_bundle.pydevd_utils import isinstance_checked, hasattr_checked, DAPGrouper
from _pydevd_bundle.pydevd_resolver import get_var_scope, MoreItems, MoreItemsRange
from typing import Optional
def str_from_providers(self, o, type_object, type_name, context: Optional[str]=None):
    provider = self._type_to_str_provider_cache.get(type_object)
    if provider is self.NO_PROVIDER:
        return None
    if provider is not None:
        return self._get_str_from_provider(provider, o, context)
    if not self._initialized:
        self._initialize()
    for provider in self._str_providers:
        if provider.can_provide(type_object, type_name):
            self._type_to_str_provider_cache[type_object] = provider
            try:
                return self._get_str_from_provider(provider, o, context)
            except:
                pydev_log.exception('Error when getting str with custom provider: %s.' % (provider,))
    self._type_to_str_provider_cache[type_object] = self.NO_PROVIDER
    return None