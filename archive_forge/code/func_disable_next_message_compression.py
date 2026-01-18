import abc
from typing import Generic, Iterable, Mapping, NoReturn, Optional, Sequence
import grpc
from ._metadata import Metadata
from ._typing import DoneCallbackType
from ._typing import MetadataType
from ._typing import RequestType
from ._typing import ResponseType
@abc.abstractmethod
def disable_next_message_compression(self) -> None:
    """Disables compression for the next response message.

        This method will override any compression configuration set during
        server creation or set on the call.
        """