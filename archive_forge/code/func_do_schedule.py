from concurrent import futures
import weakref
from automaton import machines
from oslo_utils import timeutils
from taskflow import logging
from taskflow import states as st
from taskflow.types import failure
from taskflow.utils import iter_utils
def do_schedule(next_nodes):
    with self._storage.lock.write_lock():
        return self._scheduler.schedule(sorted(next_nodes, key=lambda node: getattr(node, 'priority', 0), reverse=True))