from _pydevd_bundle import pydevd_utils
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info
from _pydevd_bundle.pydevd_comm_constants import CMD_STEP_INTO, CMD_THREAD_SUSPEND
from _pydevd_bundle.pydevd_constants import PYTHON_SUSPEND, STATE_SUSPEND, get_thread_id, STATE_RUN
from _pydev_bundle._pydev_saved_modules import threading
from _pydev_bundle import pydev_log
def resume_threads(thread_id, except_thread=None):
    pydev_log.info('Resuming threads: %s (except thread: %s)', thread_id, except_thread)
    threads = []
    if thread_id == '*':
        threads = pydevd_utils.get_non_pydevd_threads()
    elif thread_id.startswith('__frame__:'):
        pydev_log.critical("Can't make tasklet run: %s", thread_id)
    else:
        threads = [pydevd_find_thread_by_id(thread_id)]
    for t in threads:
        if t is None or t is except_thread:
            pydev_log.info('Skipped resuming thread: %s', t)
            continue
        internal_run_thread(t, set_additional_thread_info=set_additional_thread_info)