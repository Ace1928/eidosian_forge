import os
import signal
import threading
import time
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from oslo_utils import strutils
from os_brick import exception
from os_brick import privileged
def custom_execute(*cmd, **kwargs):
    """Custom execute with additional functionality on top of Oslo's.

    Additional features are timeouts and exponential backoff retries.

    The exponential backoff retries replaces standard Oslo random sleep times
    that range from 200ms to 2seconds when attempts is greater than 1, but it
    is disabled if delay_on_retry is passed as a parameter.

    Exponential backoff is controlled via interval and backoff_rate parameters,
    just like the os_brick.utils.retry decorator.

    To use the timeout mechanism to stop the subprocess with a specific signal
    after a number of seconds we must pass a non-zero timeout value in the
    call.

    When using multiple attempts and timeout at the same time the method will
    only raise the timeout exception to the caller if the last try timeouts.

    Timeout mechanism is controlled with timeout, signal, and raise_timeout
    parameters.

    :param interval: The multiplier
    :param backoff_rate: Base used for the exponential backoff
    :param timeout: Timeout defined in seconds
    :param signal: Signal to use to stop the process on timeout
    :param raise_timeout: Raise and exception on timeout or return error as
                          stderr.  Defaults to raising if check_exit_code is
                          not False.
    :returns: Tuple with stdout and stderr
    """
    shared_data = [0, None, None]

    def on_timeout(proc):
        sanitized_cmd = strutils.mask_password(' '.join(cmd))
        LOG.warning('Stopping %(cmd)s with signal %(signal)s after %(time)ss.', {'signal': sig_end, 'cmd': sanitized_cmd, 'time': timeout})
        shared_data[2] = proc
        proc.send_signal(sig_end)

    def on_execute(proc):
        if on_execute_call:
            on_execute_call(proc)
        if shared_data[0] and interval:
            exp = backoff_rate ** shared_data[0]
            wait_for = max(0, interval * exp)
            LOG.debug('Sleeping for %s seconds', wait_for)
            time.sleep(wait_for)
        shared_data[0] += 1
        if timeout:
            shared_data[2] = None
            shared_data[1] = threading.Timer(timeout, on_timeout, (proc,))
            shared_data[1].start()

    def on_completion(proc):
        if shared_data[1]:
            shared_data[1].cancel()
        if on_completion_call:
            on_completion_call(proc)
    if 'delay_on_retry' in kwargs:
        interval = None
    else:
        kwargs['delay_on_retry'] = False
        interval = kwargs.pop('interval', 1)
        backoff_rate = kwargs.pop('backoff_rate', 2)
    timeout = kwargs.pop('timeout', 600)
    sig_end = kwargs.pop('signal', signal.SIGTERM)
    default_raise_timeout = kwargs.get('check_exit_code', True)
    raise_timeout = kwargs.pop('raise_timeout', default_raise_timeout)
    on_execute_call = kwargs.pop('on_execute', None)
    on_completion_call = kwargs.pop('on_completion', None)
    try:
        return putils.execute(*cmd, on_execute=on_execute, on_completion=on_completion, **kwargs)
    except putils.ProcessExecutionError:
        proc = shared_data[2]
        if proc:
            sanitized_cmd = strutils.mask_password(' '.join(cmd))
            msg = 'Time out on proc %(pid)s after waiting %(time)s seconds when running %(cmd)s' % {'pid': proc.pid, 'time': timeout, 'cmd': sanitized_cmd}
            LOG.debug(msg)
            if raise_timeout:
                raise exception.ExecutionTimeout(stdout='', stderr=msg, cmd=sanitized_cmd)
            return ('', msg)
        raise