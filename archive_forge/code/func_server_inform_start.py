import queue
import socket
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional
from wandb.proto import wandb_server_pb2 as spb
from wandb.sdk.internal.settings_static import SettingsStatic
from ..lib import tracelog
from ..lib.sock_client import SockClient, SockClientClosedError
from .streams import StreamMux
def server_inform_start(self, sreq: 'spb.ServerRequest') -> None:
    request = sreq.inform_start
    stream_id = request._info.stream_id
    settings = SettingsStatic(request.settings)
    self._mux.update_stream(stream_id, settings=settings)
    self._mux.start_stream(stream_id)