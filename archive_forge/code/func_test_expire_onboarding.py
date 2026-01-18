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
def test_expire_onboarding(self):
    manager = self.mturk_manager
    manager.force_expire_hit = mock.MagicMock()
    self.agent_2.set_status(AssignState.STATUS_ONBOARDING)
    manager._expire_onboarding_pool()
    manager.force_expire_hit.assert_called_once_with(self.agent_2.worker_id, self.agent_2.assignment_id)