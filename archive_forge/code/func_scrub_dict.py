from sentry_sdk.utils import (
from sentry_sdk._compat import string_types
from sentry_sdk._types import TYPE_CHECKING
def scrub_dict(self, d):
    """
        If a dictionary is passed to this method, the method scrubs the dictionary of any
        sensitive data. The method calls itself recursively on any nested dictionaries (
        including dictionaries nested in lists) if self.recursive is True.
        This method does nothing if the parameter passed to it is not a dictionary.
        """
    if not isinstance(d, dict):
        return
    for k, v in d.items():
        if isinstance(k, string_types) and cast(str, k).lower() in self.denylist:
            d[k] = AnnotatedValue.substituted_because_contains_sensitive_data()
        elif self.recursive:
            self.scrub_dict(v)
            self.scrub_list(v)