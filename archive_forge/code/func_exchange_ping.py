from concurrent import futures
import enum
import errno
import io
import logging as pylogging
import os
import platform
import socket
import subprocess
import sys
import tempfile
import threading
import eventlet
from eventlet import patcher
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_privsep._i18n import _
from oslo_privsep import capabilities
from oslo_privsep import comm
def exchange_ping(self):
    try:
        reply = self.send_recv((comm.Message.PING.value,))
        success = reply[0] == comm.Message.PONG
    except Exception as e:
        self.log.exception('Error while sending initial PING to privsep: %s', e)
        success = False
    if not success:
        msg = _('Privsep daemon failed to start')
        self.log.critical(msg)
        raise FailedToDropPrivileges(msg)