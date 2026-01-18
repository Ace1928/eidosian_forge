from unittest.mock import MagicMock
from google.api_core.exceptions import Aborted, GoogleAPICallError
from cloudsdk.google.protobuf.any_pb2 import Any  # pytype: disable=pyi-error
from google.rpc.error_details_pb2 import ErrorInfo
from google.rpc.status_pb2 import Status
import grpc
from grpc_status import rpc_status
def make_call_without_metadata(status_pb: Status) -> grpc.Call:
    mock_call = make_call(status_pb)
    mock_call.trailing_metadata.return_value = None
    return mock_call