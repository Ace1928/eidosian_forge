import logging
import time
from typing import Any, AnyStr, Callable, Optional, Union
import grpc
from grpc._cython import cygrpc
from grpc._typing import DeserializingFunction
from grpc._typing import SerializingFunction
def fully_qualified_method(group: str, method: str) -> str:
    return '/{}/{}'.format(group, method)