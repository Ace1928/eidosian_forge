import collections
import contextlib
import logging
import os
import socket
import threading
from oslo_concurrency import processutils
from oslo_config import cfg
from glance_store import exceptions
from glance_store.i18n import _LE, _LW
def remove_attachment(self, vol_name, host):
    self.attachments.remove((vol_name, host))