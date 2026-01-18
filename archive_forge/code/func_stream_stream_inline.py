import collections
from grpc.framework.common import cardinality
from grpc.framework.common import style
from grpc.framework.foundation import stream  # pylint: disable=unused-import
from grpc.framework.interfaces.face import face
def stream_stream_inline(behavior):
    """Creates an face.MethodImplementation for the given behavior.

    Args:
      behavior: The implementation of a stream-stream RPC method as a callable
        value that takes an iterator of request values and an
        face.ServicerContext object and returns an iterator of response values.

    Returns:
      An face.MethodImplementation derived from the given behavior.
    """
    return _MethodImplementation(cardinality.Cardinality.STREAM_STREAM, style.Service.INLINE, None, None, None, behavior, None, None, None, None)