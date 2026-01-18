from __future__ import annotations
import atexit
import os
import sys
import debugpy
from debugpy import adapter, common, launcher
from debugpy.common import json, log, messaging, sockets
from debugpy.adapter import clients, components, launchers, servers, sessions
def notify_of_subprocess(self, conn):
    log.info('{1} is a subprocess of {0}.', self, conn)
    with self.session:
        if self.start_request is None or conn in self.known_subprocesses:
            return
        if 'processId' in self.start_request.arguments:
            log.warning('Not reporting subprocess for {0}, because the parent process was attached to using "processId" rather than "port".', self.session)
            return
        log.info('Notifying {0} about {1}.', self, conn)
        body = dict(self.start_request.arguments)
        self.known_subprocesses.add(conn)
        self.session.notify_changed()
    for key in ('processId', 'listen', 'preLaunchTask', 'postDebugTask', 'request', 'restart'):
        body.pop(key, None)
    body['name'] = 'Subprocess {0}'.format(conn.pid)
    body['subProcessId'] = conn.pid
    for key in ('args', 'processName', 'pythonArgs'):
        body.pop(key, None)
    host = body.pop('host', None)
    port = body.pop('port', None)
    if 'connect' not in body:
        body['connect'] = {}
    if 'host' not in body['connect']:
        body['connect']['host'] = host if host is not None else '127.0.0.1'
    if 'port' not in body['connect']:
        if port is None:
            _, port = listener.getsockname()
        body['connect']['port'] = port
    if self.capabilities['supportsStartDebuggingRequest']:
        self.channel.request('startDebugging', {'request': 'attach', 'configuration': body})
    else:
        body['request'] = 'attach'
        self.channel.send_event('debugpyAttach', body)