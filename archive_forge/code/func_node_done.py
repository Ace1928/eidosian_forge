from collections.abc import Mapping
import hashlib
import queue
import string
import threading
import time
import typing as ty
import keystoneauth1
from keystoneauth1 import adapter as ks_adapter
from keystoneauth1 import discover
from openstack import _log
from openstack import exceptions
def node_done(self, node):
    """Mark node as "processed" and put following items into the queue"""
    self._done.add(node)
    for v in self._graph[node]:
        self._run_in_degree[v] -= 1
        if self._run_in_degree[v] == 0:
            self._queue.put(v)