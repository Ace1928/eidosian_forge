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
def send_alive(self):
    data = {'hit_id': self.hit_id, 'assignment_id': self.assignment_id, 'worker_id': self.worker_id, 'conversation_id': self.conversation_id}
    return self.build_and_send_packet(data_model.AGENT_ALIVE, data)