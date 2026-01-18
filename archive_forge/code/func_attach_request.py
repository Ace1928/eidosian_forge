from __future__ import annotations
import atexit
import os
import sys
import debugpy
from debugpy import adapter, common, launcher
from debugpy.common import json, log, messaging, sockets
from debugpy.adapter import clients, components, launchers, servers, sessions
@_start_message_handler
def attach_request(self, request):
    if self.session.no_debug:
        raise request.isnt_valid('"noDebug" is not supported for "attach"')
    host = request('host', str, optional=True)
    port = request('port', int, optional=True)
    listen = request('listen', dict, optional=True)
    connect = request('connect', dict, optional=True)
    pid = request('processId', (int, str), optional=True)
    sub_pid = request('subProcessId', int, optional=True)
    on_terminate = request('onTerminate', bool, optional=True)
    if on_terminate:
        self._forward_terminate_request = on_terminate == 'KeyboardInterrupt'
    if host != () or port != ():
        if listen != ():
            raise request.isnt_valid('"listen" and "host"/"port" are mutually exclusive')
        if connect != ():
            raise request.isnt_valid('"connect" and "host"/"port" are mutually exclusive')
    if listen != ():
        if connect != ():
            raise request.isnt_valid('"listen" and "connect" are mutually exclusive')
        if pid != ():
            raise request.isnt_valid('"listen" and "processId" are mutually exclusive')
        if sub_pid != ():
            raise request.isnt_valid('"listen" and "subProcessId" are mutually exclusive')
    if pid != () and sub_pid != ():
        raise request.isnt_valid('"processId" and "subProcessId" are mutually exclusive')
    if listen != ():
        if servers.is_serving():
            raise request.isnt_valid('Multiple concurrent "listen" sessions are not supported')
        host = listen('host', '127.0.0.1')
        port = listen('port', int)
        adapter.access_token = None
        self.restart_requested = request('restart', False)
        host, port = servers.serve(host, port)
    else:
        if not servers.is_serving():
            servers.serve()
        host, port = servers.listener.getsockname()
    if pid != ():
        if not isinstance(pid, int):
            try:
                pid = int(pid)
            except Exception:
                raise request.isnt_valid('"processId" must be parseable as int')
        debugpy_args = request('debugpyArgs', json.array(str))

        def on_output(category, output):
            self.channel.send_event('output', {'category': category, 'output': output})
        try:
            servers.inject(pid, debugpy_args, on_output)
        except Exception as e:
            log.swallow_exception()
            self.session.finalize('Error when trying to attach to PID:\n%s' % (str(e),))
            return
        timeout = common.PROCESS_SPAWN_TIMEOUT
        pred = lambda conn: conn.pid == pid
    elif sub_pid == ():
        pred = lambda conn: True
        timeout = common.PROCESS_SPAWN_TIMEOUT if listen == () else None
    else:
        pred = lambda conn: conn.pid == sub_pid
        timeout = 0
    self.channel.send_event('debugpyWaitingForServer', {'host': host, 'port': port})
    conn = servers.wait_for_connection(self.session, pred, timeout)
    if conn is None:
        if sub_pid != ():
            request.respond({})
            self.session.finalize('No known subprocess with "subProcessId":{0}'.format(sub_pid))
            return
        raise request.cant_handle('Timed out waiting for debug server to connect.' if timeout else 'There is no debug server connected to this adapter.', sub_pid)
    try:
        conn.attach_to_session(self.session)
    except ValueError:
        request.cant_handle('{0} is already being debugged.', conn)