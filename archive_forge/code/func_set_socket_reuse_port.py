import asyncio
import inspect
import json
import logging
import pickle
import socket
from typing import Any, List, Optional, Type
import starlette
from fastapi.encoders import jsonable_encoder
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from uvicorn.config import Config
from uvicorn.lifespan.on import LifespanOn
from ray._private.pydantic_compat import IS_PYDANTIC_2
from ray.actor import ActorHandle
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.utils import serve_encoders
from ray.serve.exceptions import RayServeException
def set_socket_reuse_port(sock: socket.socket) -> bool:
    """Mutate a socket object to allow multiple process listening on the same port.

    Returns:
        success: whether the setting was successful.
    """
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, 'SO_REUSEPORT'):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        else:
            sock.setsockopt(socket.SOL_SOCKET, 15, 1)
        return True
    except Exception as e:
        logger.debug(f'Setting SO_REUSEPORT failed because of {e}. SO_REUSEPORT is disabled.')
        return False