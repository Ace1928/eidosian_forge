from typing import Generic, Iterator, Optional, TypeVar
import collections
import functools
import warnings
import grpc
from google.api_core import exceptions
import google.auth
import google.auth.credentials
import google.auth.transport.grpc
import google.auth.transport.requests
import cloudsdk.google.protobuf
@functools.wraps(callable_)
def error_remapped_callable(*args, **kwargs):
    try:
        result = callable_(*args, **kwargs)
        prefetch_first = getattr(callable_, '_prefetch_first_result_', True)
        return _StreamingResponseIterator(result, prefetch_first_result=prefetch_first)
    except grpc.RpcError as exc:
        raise exceptions.from_grpc_error(exc) from exc