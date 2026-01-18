import datetime
import six
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.oauth2 import sts
def add_rule(self, rule):
    """Adds a single access boundary rule to the existing rules.

        Args:
            rule (google.auth.downscoped.AccessBoundaryRule): The access boundary rule,
                limiting the access that a downscoped credential will have, to be added to
                the existing rules.
        Raises:
            InvalidType: If any of the rules are not a valid type.
            InvalidValue: If the provided rules exceed the maximum allowed.
        """
    if len(self.rules) == _MAX_ACCESS_BOUNDARY_RULES_COUNT:
        raise exceptions.InvalidValue('Credential access boundary rules can have a maximum of {} rules.'.format(_MAX_ACCESS_BOUNDARY_RULES_COUNT))
    if not isinstance(rule, AccessBoundaryRule):
        raise exceptions.InvalidType("The provided rule does not contain a valid 'google.auth.downscoped.AccessBoundaryRule'.")
    self._rules.append(rule)