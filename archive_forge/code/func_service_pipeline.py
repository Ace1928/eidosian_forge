import collections
import sys
import types
from typing import Any, Callable, Optional, Sequence, Tuple, Union
import grpc
from ._typing import DeserializingFunction
from ._typing import DoneCallbackType
from ._typing import MetadataType
from ._typing import RequestIterableType
from ._typing import SerializingFunction
def service_pipeline(interceptors: Optional[Sequence[grpc.ServerInterceptor]]) -> Optional[_ServicePipeline]:
    return _ServicePipeline(interceptors) if interceptors else None