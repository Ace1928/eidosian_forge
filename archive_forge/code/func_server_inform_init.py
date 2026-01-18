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
def server_inform_init(self, sreq: 'spb.ServerRequest') -> None:
    request = sreq.inform_init
    stream_id = request._info.stream_id
    settings = SettingsStatic(request.settings)
    self._mux.add_stream(stream_id, settings=settings)
    iface = self._mux.get_stream(stream_id).interface
    self._clients.add_client(self._sock_client)
    iface_reader_thread = SockServerInterfaceReaderThread(clients=self._clients, iface=iface, stopped=self._stopped)
    iface_reader_thread.start()