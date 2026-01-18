from __future__ import annotations
import os
import time
import warnings
import httpx
from gradio.routes import App  # HACK: to avoid circular import # noqa: F401
from gradio.tunneling import Tunnel
def setup_tunnel(local_host: str, local_port: int, share_token: str, share_server_address: str | None) -> str:
    share_server_address = GRADIO_SHARE_SERVER_ADDRESS if share_server_address is None else share_server_address
    if share_server_address is None:
        try:
            response = httpx.get(GRADIO_API_SERVER, timeout=30)
            payload = response.json()[0]
            remote_host, remote_port = (payload['host'], int(payload['port']))
        except Exception as e:
            raise RuntimeError('Could not get share link from Gradio API Server.') from e
    else:
        remote_host, remote_port = share_server_address.split(':')
        remote_port = int(remote_port)
    try:
        tunnel = Tunnel(remote_host, remote_port, local_host, local_port, share_token)
        address = tunnel.start_tunnel()
        return address
    except Exception as e:
        raise RuntimeError(str(e)) from e