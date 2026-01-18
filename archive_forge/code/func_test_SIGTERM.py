import os
import signal
import time
from http.client import BadStatusLine
import pytest
import portend
import cherrypy
import cherrypy.process.servers
from cherrypy.test import helper
def test_SIGTERM(self):
    """SIGTERM should shut down the server whether daemonized or not."""
    self._require_signal_and_kill('SIGTERM')
    p = helper.CPProcess(ssl=self.scheme.lower() == 'https')
    p.write_conf(extra='test_case_name: "test_SIGTERM"')
    p.start(imports='cherrypy.test._test_states_demo')
    os.kill(p.get_pid(), signal.SIGTERM)
    p.join()
    if os.name in ['posix']:
        p = helper.CPProcess(ssl=self.scheme.lower() == 'https', wait=True, daemonize=True)
        p.write_conf(extra='test_case_name: "test_SIGTERM_2"')
        p.start(imports='cherrypy.test._test_states_demo')
        os.kill(p.get_pid(), signal.SIGTERM)
        p.join()