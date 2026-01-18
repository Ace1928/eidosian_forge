from __future__ import annotations
import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Any, Final
from streamlit import (
from streamlit.config import CONFIG_FILENAMES
from streamlit.git_util import MIN_GIT_VERSION, GitRepo
from streamlit.logger import get_logger
from streamlit.source_util import invalidate_pages_cache
from streamlit.watcher import report_watchdog_availability, watch_dir, watch_file
from streamlit.web.server import Server, server_address_is_unix_socket, server_util
def maybe_open_browser():
    if config.get_option('server.headless'):
        return
    if config.is_manually_set('browser.serverAddress'):
        addr = config.get_option('browser.serverAddress')
    elif config.is_manually_set('server.address'):
        if server_address_is_unix_socket():
            return
        addr = config.get_option('server.address')
    else:
        addr = 'localhost'
    util.open_browser(server_util.get_url(addr))