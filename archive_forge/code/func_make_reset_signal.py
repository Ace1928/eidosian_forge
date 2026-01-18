from unittest.mock import MagicMock
from google.api_core.exceptions import Aborted, GoogleAPICallError
from cloudsdk.google.protobuf.any_pb2 import Any  # pytype: disable=pyi-error
from google.rpc.error_details_pb2 import ErrorInfo
from google.rpc.status_pb2 import Status
import grpc
from grpc_status import rpc_status
def make_reset_signal() -> GoogleAPICallError:
    any = Any()
    any.Pack(ErrorInfo(reason='RESET', domain='pubsublite.googleapis.com'))
    status_pb = Status(code=10, details=[any])
    return Aborted('', response=make_call(status_pb))