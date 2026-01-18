import asyncio
import subprocess
import sys
import pytest
from tornado.queues import Queue
from jupyter_lsp.stdio import LspStdIoReader
from time import sleep
@pytest.fixture
def communicator_spawner(tmp_path):
    return CommunicatorSpawner(tmp_path)