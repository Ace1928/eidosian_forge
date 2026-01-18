import json
import os
import socket
import subprocess
import sys
import time
import uuid
from typing import Generator, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import pytest
from .conftest import KNOWN_SERVERS, extra_node_roots
@pytest.mark.parametrize('route', WS_ROUTES)
def test_auth_websocket(route: str, a_server_url_and_token: Tuple[str, str]) -> None:
    """Verify a WebSocket does not provide access to an unauthenticated user."""
    verify_response(a_server_url_and_token[0], route)