from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import contextlib
import sys
import threading
import time
import six
import tensorflow as tf
from tensorflow_estimator.python.estimator.tools import analytics
def record_error(self, source, exc_info, session=None):
    """Report an exception from the given source.

    If a session is passed, a timer will be registered to close it after a few
    seconds.  This is necessary to ensure the main training loop does not hang
    if an infeed/oufeed error occurs.  We sleep a few seconds to allow a more
    interesting error from another thread to propagate.

    Args:
      source: string, source of the error
      exc_info: Output from `sys.exc_info` (type, value, traceback)
      session: Session to close after delay.
    """
    _, value, _ = exc_info
    if isinstance(value, _IGNORED_ERRORS):
        return
    self._errors[source] = exc_info
    try:
        if value.op.type == _CHECK_NUMERIC_OP_NAME:
            analytics.track_numerical_issues(exc_info)
            return
    except AttributeError as _:
        pass
    if session is not None and self._session_cancel_timer is None:

        def _cancel_session():
            time.sleep(5)
            tf.compat.v1.logging.error('Closing session due to error %s' % value)
            try:
                session.close()
            except:
                tf.compat.v1.logging.error('\n\n\nFailed to close session after error.Other threads may hang.\n\n\n')
        self._session_cancel_timer = threading.Thread(target=_cancel_session)
        self._session_cancel_timer.daemon = True
        self._session_cancel_timer.start()