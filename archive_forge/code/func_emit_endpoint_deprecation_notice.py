from __future__ import annotations
from typing import Final
from urllib.parse import urljoin
import tornado.web
from streamlit import config, net_util, url_util
def emit_endpoint_deprecation_notice(handler: tornado.web.RequestHandler, new_path: str) -> None:
    """
    Emits the warning about deprecation of HTTP endpoint in the HTTP header.
    """
    handler.set_header('Deprecation', True)
    new_url = urljoin(f'{handler.request.protocol}://{handler.request.host}', new_path)
    handler.set_header('Link', f'<{new_url}>; rel="alternate"')