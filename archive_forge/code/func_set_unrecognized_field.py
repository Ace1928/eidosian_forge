import types
import weakref
import six
from apitools.base.protorpclite import util
def set_unrecognized_field(self, key, value, variant):
    """Set an unrecognized field, used when decoding a message.

        Args:
          key: The name or number used to refer to this unknown value.
          value: The value of the field.
          variant: Type information needed to interpret the value or re-encode
            it.

        Raises:
          TypeError: If the variant is not an instance of messages.Variant.
        """
    if not isinstance(variant, Variant):
        raise TypeError('Variant type %s is not valid.' % variant)
    self.__unrecognized_fields[key] = (value, variant)