from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
def remove_toleration(current, key_value, effect):
    """Removes a toleration from the current deployment configuration.

  A toleration must match exactly to be removed - it is not enough to match the
  key, or even key-value. The effect must also match the toleration being
  removed.

  Args:
    current: the deployment configuration object being modified.
    key_value: the key-and-optional-value string specifying the toleration key
      and value.
    effect: Optional. If included, will set the effect value on the toleration.

  Returns:
    The modified deployment configuration object.
  """
    current_tolerations = current.podTolerations
    key, value, operator = _parse_key_value(key_value)

    def match(toleration):
        return toleration.key == key and toleration.value == value and (toleration.operator == operator) and (toleration.effect == effect)
    current.podTolerations = [t for t in current_tolerations if not match(t)]
    return current