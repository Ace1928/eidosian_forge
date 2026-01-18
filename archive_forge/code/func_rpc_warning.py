import uuid
import warnings
from inspect import signature
from typing import Callable, Optional, Tuple, Union, List, Any
import rpcq.messages
def rpc_warning(warning: warnings.WarningMessage) -> rpcq.messages.RPCWarning:
    return rpcq.messages.RPCWarning(body=str(warning.message), kind=str(warning.category.__name__))