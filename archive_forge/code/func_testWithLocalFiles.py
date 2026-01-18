from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import threading
import six
from six.moves import queue as Queue
from six.moves import range
from gslib.name_expansion import SeekAheadNameExpansionIterator
from gslib.seek_ahead_thread import SeekAheadResult
from gslib.seek_ahead_thread import SeekAheadThread
import gslib.tests.testcase as testcase
from gslib.ui_controller import UIController
from gslib.ui_controller import UIThread
from gslib.utils import constants
from gslib.utils import parallelism_framework_util
from gslib.utils import unit_util
def testWithLocalFiles(self):
    """Tests SeekAheadThread with an actual directory."""
    tmpdir = self.CreateTempDir()
    num_files = 5
    total_size = 0
    for i in range(num_files):
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj%s' % str(i), contents=b'a' * i)
        total_size += i
    seek_ahead_iterator = SeekAheadNameExpansionIterator('cp', 0, None, [tmpdir], True)
    cancel_event = threading.Event()
    status_queue = Queue.Queue()
    stream = six.StringIO()
    ui_controller = UIController()
    ui_thread = UIThread(status_queue, stream, ui_controller)
    seek_ahead_thread = SeekAheadThread(seek_ahead_iterator, cancel_event, status_queue)
    seek_ahead_thread.join(self.thread_wait_time)
    status_queue.put(_ZERO_TASKS_TO_DO_ARGUMENT)
    ui_thread.join(self.thread_wait_time)
    if seek_ahead_thread.is_alive():
        seek_ahead_thread.terminate = True
        self.fail('SeekAheadThread is still alive.')
    message = stream.getvalue()
    if not message:
        self.fail('Status queue empty but SeekAheadThread should have posted summary message')
    self.assertEqual(message, 'Estimated work for this command: objects: %s, total size: %s\n' % (num_files, unit_util.MakeHumanReadable(total_size)))