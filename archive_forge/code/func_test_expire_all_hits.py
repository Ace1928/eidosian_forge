import unittest
import os
import time
import json
import threading
import pickle
from unittest import mock
from parlai.mturk.core.dev.worker_manager import WorkerManager
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
from parlai.mturk.core.dev.worlds import MTurkOnboardWorld
from parlai.mturk.core.dev.mturk_manager import MTurkManager
from parlai.mturk.core.dev.socket_manager import SocketManager, Packet
from parlai.core.params import ParlaiParser
from websocket_server import WebsocketServer
import parlai.mturk.core.dev.mturk_manager as MTurkManagerFile
import parlai.mturk.core.dev.data_model as data_model
def test_expire_all_hits(self):
    manager = self.mturk_manager
    incomplete_1 = 'incomplete_1'
    incomplete_2 = 'incomplete_2'
    MTurkManagerFile.mturk_utils.expire_hit = mock.MagicMock()
    manager.hit_id_list = [incomplete_1, incomplete_2]
    manager.expire_all_unassigned_hits()
    expire_calls = MTurkManagerFile.mturk_utils.expire_hit.call_args_list
    self.assertEqual(len(expire_calls), 2)
    for hit in [incomplete_1, incomplete_2]:
        found = False
        for expire_call in expire_calls:
            if expire_call[0][1] == hit:
                found = True
                break
        self.assertTrue(found)