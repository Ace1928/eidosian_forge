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
def testCancellation(self):
    """Tests cancellation of SeekAheadThread."""

    class TrackingCancellationIterator(object):
        """Yields dummy results and sends cancellation after some # of yields."""

        def __init__(self, num_iterations, num_iterations_before_cancel, cancel_event):
            """Initializes the iterator.

        Args:
          num_iterations: Total number of results to yield.
          num_iterations_before_cancel: Set cancel event before yielding
              on the given iteration.
          cancel_event: threading.Event() to signal SeekAheadThread to stop.
        """
            self.num_iterations_before_cancel = num_iterations_before_cancel
            self.iterated_results = 0
            self.num_iterations = num_iterations
            self.cancel_issued = False
            self.cancel_event = cancel_event

        def __iter__(self):
            while self.iterated_results < self.num_iterations:
                if not self.cancel_issued and self.iterated_results >= self.num_iterations_before_cancel:
                    self.cancel_event.set()
                    self.cancel_issued = True
                yield SeekAheadResult()
                self.iterated_results += 1
    noplp = constants.NUM_OBJECTS_PER_LIST_PAGE
    for num_iterations, num_iterations_before_cancel, expected_iterations in ((noplp, 0, 0), (noplp + 1, 1, noplp), (noplp + 1, noplp, noplp), (noplp * 2 + 1, noplp + 1, noplp * 2), (2, 1, 2), (noplp, 1, noplp), (noplp * 2, noplp + 1, noplp * 2)):
        cancel_event = threading.Event()
        status_queue = Queue.Queue()
        stream = six.StringIO()
        ui_controller = UIController()
        ui_thread = UIThread(status_queue, stream, ui_controller)
        seek_ahead_iterator = TrackingCancellationIterator(num_iterations, num_iterations_before_cancel, cancel_event)
        seek_ahead_thread = SeekAheadThread(seek_ahead_iterator, cancel_event, status_queue)
        seek_ahead_thread.join(self.thread_wait_time)
        status_queue.put(_ZERO_TASKS_TO_DO_ARGUMENT)
        ui_thread.join(self.thread_wait_time)
        if seek_ahead_thread.is_alive():
            seek_ahead_thread.terminate = True
            self.fail('Cancellation issued after %s iterations, but SeekAheadThread is still alive.' % num_iterations_before_cancel)
        self.assertEqual(expected_iterations, seek_ahead_iterator.iterated_results, 'Cancellation issued after %s iterations, SeekAheadThread iterated %s results, expected: %s results.' % (num_iterations_before_cancel, seek_ahead_iterator.iterated_results, expected_iterations))
        message = stream.getvalue()
        if message:
            self.fail('Status queue should be empty but contains message: %s' % message)