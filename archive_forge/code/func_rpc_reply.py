import uuid
import warnings
from inspect import signature
from typing import Callable, Optional, Tuple, Union, List, Any
import rpcq.messages
def rpc_reply(id: Union[str, int], result: Optional[object], warnings: Optional[List[Warning]]=None) -> rpcq.messages.RPCReply:
    """
    Create RPC reply

    :param str|int id: Request ID
    :param result: Result
    :param warnings: List of warnings to attach to the message
    :return: JSON RPC formatted dict
    """
    warnings = warnings or []
    return rpcq.messages.RPCReply(jsonrpc='2.0', id=id, result=result, warnings=[rpc_warning(warning) for warning in warnings])