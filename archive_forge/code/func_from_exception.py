from typing import Optional
from wandb.proto import wandb_internal_pb2 as pb
from . import AuthenticationError, CommError, Error, UnsupportedError, UsageError
from_exception_map = {v: k for k, v in to_exception_map.items()}
@classmethod
def from_exception(cls, exc: Error) -> 'pb.ErrorInfo':
    """Convert an wandb error to a protobuf error message.

        Args:
            exc: The exception to convert.

        Returns:
            The corresponding protobuf error message.
        """
    if not isinstance(exc, Error):
        raise ValueError('exc must be a subclass of wandb.errors.Error')
    code = None
    for subclass in type(exc).__mro__:
        if subclass in from_exception_map:
            code = from_exception_map[subclass]
            break
    return pb.ErrorInfo(code=code, message=str(exc))