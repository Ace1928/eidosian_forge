import unittest
import time
import uuid
from unittest import mock
from parlai.mturk.core.socket_manager import Packet, SocketManager
from parlai.mturk.core.shared_utils import AssignState
import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.shared_utils as shared_utils
import threading
from websocket_server import WebsocketServer
import json
def on_hb(*args):
    nonlocal incoming_hb, hb_count
    incoming_hb = args[0]
    hb_count += 1