import contextlib
import threading
from kazoo.protocol import paths as k_paths
from kazoo.recipe import watchers
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
import testtools
from zake import fake_client
from zake import utils as zake_utils
from taskflow import exceptions as excp
from taskflow.jobs.backends import impl_zookeeper
from taskflow import states
from taskflow import test
from taskflow.test import mock
from taskflow.tests.unit.jobs import base
from taskflow.tests import utils as test_utils
from taskflow.types import entity
from taskflow.utils import kazoo_utils
from taskflow.utils import misc
from taskflow.utils import persistence_utils as p_utils
def test_board_iter(self):
    with base.connect_close(self.board):
        it = self.board.iterjobs()
        self.assertEqual(self.board, it.board)
        self.assertFalse(it.only_unclaimed)
        self.assertFalse(it.ensure_fresh)