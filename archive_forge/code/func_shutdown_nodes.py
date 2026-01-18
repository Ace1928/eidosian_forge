import errno
import os
import shlex
import signal
import sys
from collections import OrderedDict, UserList, defaultdict
from functools import partial
from subprocess import Popen
from time import sleep
from kombu.utils.encoding import from_utf8
from kombu.utils.objects import cached_property
from celery.platforms import IS_WINDOWS, Pidfile, signal_name
from celery.utils.nodenames import gethostname, host_format, node_format, nodesplit
from celery.utils.saferepr import saferepr
def shutdown_nodes(self, nodes, sig=signal.SIGTERM, retry=None):
    P = set(nodes)
    maybe_call(self.on_stopping_preamble, nodes)
    to_remove = set()
    for node in P:
        maybe_call(self.on_send_signal, node, signal_name(sig))
        if not node.send(sig, self.on_node_signal_dead):
            to_remove.add(node)
            yield node
    P -= to_remove
    if retry:
        maybe_call(self.on_still_waiting_for, P)
        its = 0
        while P:
            to_remove = set()
            for node in P:
                its += 1
                maybe_call(self.on_still_waiting_progress, P)
                if not node.alive():
                    maybe_call(self.on_node_shutdown_ok, node)
                    to_remove.add(node)
                    yield node
                    maybe_call(self.on_still_waiting_for, P)
                    break
            P -= to_remove
            if P and (not its % len(P)):
                sleep(float(retry))
        maybe_call(self.on_still_waiting_end)