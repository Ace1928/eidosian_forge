import os
import sys
import time
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, threads
from twisted.python import failure, log, threadable, threadpool
from twisted.trial.unittest import TestCase
import time
import %(reactor)s
from twisted.internet import reactor
def programFinished(result):
    out, err, reason = result
    if reason.check(error.ProcessTerminated):
        self.fail(f'Process did not exit cleanly (out: {out} err: {err})')
    if err:
        log.msg(f'Unexpected output on standard error: {err}')
    self.assertFalse(out, f'Expected no output, instead received:\n{out}')