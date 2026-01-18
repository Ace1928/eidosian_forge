import re
from fqdn._compat import cached_property
@cached_property
def is_valid_absolute(self):
    """
        True for a fully-qualified domain name (FQDN) that is RFC
        preferred-form compliant and ends with a `.`.

        With relative FQDNS in DNS lookups, the current hosts domain name or
        search domains may be appended.
        """
    return self._fqdn.endswith('.') and self.is_valid