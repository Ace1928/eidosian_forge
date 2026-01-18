from __future__ import annotations
import json
import logging
import os
from typing import TYPE_CHECKING, Any
import tornado.websocket
from tornado import gen
from tornado.concurrent import run_on_executor
def on_pty_died(self) -> None:
    """Terminal closed: tell the frontend, and close the socket."""
    self.send_json_message(['disconnect', 1])
    self.close()
    self.terminal = None