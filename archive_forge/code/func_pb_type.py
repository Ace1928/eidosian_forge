from enum import EnumMeta
from cloudsdk.google.protobuf import descriptor_pb2
from cloudsdk.google.protobuf.internal.enum_type_wrapper import EnumTypeWrapper
from proto.primitives import ProtoType
@property
def pb_type(self):
    """Return the composite type of the field, or the primitive type if a primitive."""
    if self.enum:
        return self.enum
    if not self.message:
        return self.proto_type
    if hasattr(self.message, '_meta'):
        return self.message.pb()
    return self.message