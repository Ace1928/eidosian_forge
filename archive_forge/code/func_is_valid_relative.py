import re
from fqdn._compat import cached_property
@cached_property
def is_valid_relative(self):
    """
        True for a validated fully-qualified domain name that compiles with the
        RFC preferred-form and does not ends with a `.`.
        """
    return not self._fqdn.endswith('.') and self.is_valid