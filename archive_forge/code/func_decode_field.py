import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
def decode_field(self, field, value):
    """Decode the given JSON value.

        Args:
          field: a messages.Field for the field we're decoding.
          value: a python value we'd like to decode.

        Returns:
          A value suitable for assignment to field.
        """
    for decoder in _GetFieldCodecs(field, 'decoder'):
        result = decoder(field, value)
        value = result.value
        if result.complete:
            return value
    if isinstance(field, messages.MessageField):
        field_value = self.decode_message(field.message_type, json.dumps(value))
    elif isinstance(field, messages.EnumField):
        value = GetCustomJsonEnumMapping(field.type, json_name=value) or value
        try:
            field_value = super(_ProtoJsonApiTools, self).decode_field(field, value)
        except messages.DecodeError:
            if not isinstance(value, six.string_types):
                raise
            field_value = None
    else:
        field_value = super(_ProtoJsonApiTools, self).decode_field(field, value)
    return field_value