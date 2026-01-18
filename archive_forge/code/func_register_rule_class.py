import abc
import enum
from cloudsdk.google.protobuf import message
from cloudsdk.google.protobuf import duration_pb2
from cloudsdk.google.protobuf import timestamp_pb2
from cloudsdk.google.protobuf import field_mask_pb2
from cloudsdk.google.protobuf import struct_pb2
from cloudsdk.google.protobuf import wrappers_pb2
from proto.marshal import compat
from proto.marshal.collections import MapComposite
from proto.marshal.collections import Repeated
from proto.marshal.collections import RepeatedComposite
from proto.marshal.rules import bytes as pb_bytes
from proto.marshal.rules import stringy_numbers
from proto.marshal.rules import dates
from proto.marshal.rules import struct
from proto.marshal.rules import wrappers
from proto.marshal.rules import field_mask
from proto.primitives import ProtoType
def register_rule_class(rule_class: type):
    if not issubclass(rule_class, Rule):
        raise TypeError('Marshal rule subclasses must implement `to_proto` and `to_python` methods.')
    self._rules[proto_type] = rule_class()
    return rule_class