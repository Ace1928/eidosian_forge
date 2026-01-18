import unittest
import time
import uuid
from unittest import mock
from parlai.mturk.core.dev.socket_manager import Packet, SocketManager
from parlai.mturk.core.dev.agents import AssignState
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
import threading
from websocket_server import WebsocketServer
import json
def register_to_socket(self, ws, on_msg):
    handler = self.make_packet_handler(on_msg)
    self.ws = ws
    self.ws.handlers[self.worker_id] = handler