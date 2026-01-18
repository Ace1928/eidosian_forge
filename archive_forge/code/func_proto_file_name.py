import collections
import inspect
import logging
from cloudsdk.google.protobuf import descriptor_pb2
from cloudsdk.google.protobuf import descriptor_pool
from cloudsdk.google.protobuf import message
from cloudsdk.google.protobuf import reflection
from proto.marshal.rules.message import MessageRule
@staticmethod
def proto_file_name(name):
    return '{0}.proto'.format(name.replace('.', '/'))