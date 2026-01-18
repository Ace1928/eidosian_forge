from __future__ import annotations
import contextlib
import inspect
import io
import json
import math
import queue
import sys
import typing
import warnings
from concurrent.futures import Future
from functools import cached_property
from types import GeneratorType
from urllib.parse import unquote, urljoin
import anyio
import anyio.abc
import anyio.from_thread
from anyio.abc import ObjectReceiveStream, ObjectSendStream
from anyio.streams.stapled import StapledObjectStream
from starlette._utils import is_async_callable
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from starlette.websockets import WebSocketDisconnect
def receive_json(self, mode: typing.Literal['text', 'binary']='text') -> typing.Any:
    message = self.receive()
    self._raise_on_close(message)
    if mode == 'text':
        text = message['text']
    else:
        text = message['bytes'].decode('utf-8')
    return json.loads(text)