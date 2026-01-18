import uuid
import warnings
from inspect import signature
from typing import Callable, Optional, Tuple, Union, List, Any
import rpcq.messages
def get_safe_input(params: Union[dict, list], handler: Callable) -> Tuple[list, dict]:
    """
    Get positional or keyword arguments from JSON RPC params,
       filtering out kwargs that aren't in the function signature

    :param params: Parameters passed through JSON RPC
    :param handler: RPC handler function
    :return: args, kwargs
    """
    args, kwargs = get_input(params)
    handler_signature = signature(handler)
    kwargs = {k: v for k, v in kwargs.items() if k in handler_signature.parameters}
    return (args, kwargs)