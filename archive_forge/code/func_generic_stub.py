import threading  # pylint: disable=unused-import
import grpc
from grpc import _auth
from grpc.beta import _client_adaptations
from grpc.beta import _metadata
from grpc.beta import _server_adaptations
from grpc.beta import interfaces  # pylint: disable=unused-import
from grpc.framework.common import cardinality  # pylint: disable=unused-import
from grpc.framework.interfaces.face import face  # pylint: disable=unused-import
def generic_stub(channel, options=None):
    """Creates a face.GenericStub on which RPCs can be made.

    Args:
      channel: A Channel for use by the created stub.
      options: A StubOptions customizing the created stub.

    Returns:
      A face.GenericStub on which RPCs can be made.
    """
    effective_options = _EMPTY_STUB_OPTIONS if options is None else options
    return _client_adaptations.generic_stub(channel._channel, effective_options.host, effective_options.metadata_transformer, effective_options.request_serializers, effective_options.response_deserializers)