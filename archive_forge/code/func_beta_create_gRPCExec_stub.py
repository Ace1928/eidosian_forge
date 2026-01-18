from __future__ import absolute_import, division, print_function
import sys
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import enum_type_wrapper
import grpc
from grpc.beta import implementations as beta_implementations
from grpc.beta import interfaces as beta_interfaces
from grpc.framework.common import cardinality
from grpc.framework.interfaces.face import utilities as face_utilities
def beta_create_gRPCExec_stub(channel, host=None, metadata_transformer=None, pool=None, pool_size=None):
    request_serializers = {('IOSXRExtensibleManagabilityService.gRPCExec', 'ShowCmdJSONOutput'): ShowCmdArgs.SerializeToString, ('IOSXRExtensibleManagabilityService.gRPCExec', 'ShowCmdTextOutput'): ShowCmdArgs.SerializeToString}
    response_deserializers = {('IOSXRExtensibleManagabilityService.gRPCExec', 'ShowCmdJSONOutput'): ShowCmdJSONReply.FromString, ('IOSXRExtensibleManagabilityService.gRPCExec', 'ShowCmdTextOutput'): ShowCmdTextReply.FromString}
    cardinalities = {'ShowCmdJSONOutput': cardinality.Cardinality.UNARY_STREAM, 'ShowCmdTextOutput': cardinality.Cardinality.UNARY_STREAM}
    stub_options = beta_implementations.stub_options(host=host, metadata_transformer=metadata_transformer, request_serializers=request_serializers, response_deserializers=response_deserializers, thread_pool=pool, thread_pool_size=pool_size)
    return beta_implementations.dynamic_stub(channel, 'IOSXRExtensibleManagabilityService.gRPCExec', cardinalities, options=stub_options)