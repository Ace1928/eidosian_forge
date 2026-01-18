import contextlib
import threading
import time
from taskflow import exceptions as excp
from taskflow.persistence.backends import impl_dir
from taskflow import states
from taskflow.tests import utils as test_utils
from taskflow.utils import persistence_utils as p_utils
from taskflow.utils import threading_utils
def test_board_iter_empty(self):
    with connect_close(self.board):
        jobs_found = list(self.board.iterjobs())
        self.assertEqual([], jobs_found)