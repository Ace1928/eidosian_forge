import threading  # pylint: disable=unused-import
import grpc
from grpc import _auth
from grpc.beta import _client_adaptations
from grpc.beta import _metadata
from grpc.beta import _server_adaptations
from grpc.beta import interfaces  # pylint: disable=unused-import
from grpc.framework.common import cardinality  # pylint: disable=unused-import
from grpc.framework.interfaces.face import face  # pylint: disable=unused-import
def stub_options(host=None, request_serializers=None, response_deserializers=None, metadata_transformer=None, thread_pool=None, thread_pool_size=None):
    """Creates a StubOptions value to be passed at stub creation.

    All parameters are optional and should always be passed by keyword.

    Args:
      host: A host string to set on RPC calls.
      request_serializers: A dictionary from service name-method name pair to
        request serialization behavior.
      response_deserializers: A dictionary from service name-method name pair to
        response deserialization behavior.
      metadata_transformer: A callable that given a metadata object produces
        another metadata object to be used in the underlying communication on the
        wire.
      thread_pool: A thread pool to use in stubs.
      thread_pool_size: The size of thread pool to create for use in stubs;
        ignored if thread_pool has been passed.

    Returns:
      A StubOptions value created from the passed parameters.
    """
    return StubOptions(host, request_serializers, response_deserializers, metadata_transformer, thread_pool, thread_pool_size)