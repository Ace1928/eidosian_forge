import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def strip_doc_string(proto: google.protobuf.message.Message) -> None:
    """Empties `doc_string` field on any nested protobuf messages"""
    if not isinstance(proto, google.protobuf.message.Message):
        raise TypeError(f'proto must be an instance of {google.protobuf.message.Message}.')
    for descriptor in proto.DESCRIPTOR.fields:
        if descriptor.name == 'doc_string':
            proto.ClearField(descriptor.name)
        elif descriptor.type == descriptor.TYPE_MESSAGE:
            if descriptor.label == descriptor.LABEL_REPEATED:
                for x in getattr(proto, descriptor.name):
                    strip_doc_string(x)
            elif proto.HasField(descriptor.name):
                strip_doc_string(getattr(proto, descriptor.name))