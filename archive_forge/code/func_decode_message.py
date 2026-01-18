import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
def decode_message(self, message_type, encoded_message):
    if message_type in _CUSTOM_MESSAGE_CODECS:
        return _CUSTOM_MESSAGE_CODECS[message_type].decoder(encoded_message)
    result = _DecodeCustomFieldNames(message_type, encoded_message)
    result = super(_ProtoJsonApiTools, self).decode_message(message_type, result)
    result = _ProcessUnknownEnums(result, encoded_message)
    result = _ProcessUnknownMessages(result, encoded_message)
    return _DecodeUnknownFields(result, encoded_message)