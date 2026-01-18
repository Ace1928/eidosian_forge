import os
import signal
import time
from http.client import BadStatusLine
import pytest
import portend
import cherrypy
import cherrypy.process.servers
from cherrypy.test import helper
@pytest.mark.xfail('sys.platform == "Darwin" and sys.version_info > (3, 7) and os.environ["TRAVIS"]', reason='https://github.com/cherrypy/cherrypy/issues/1693')
def test_4_Autoreload(self):
    if engine.state != engine.states.EXITING:
        engine.exit()
    p = helper.CPProcess(ssl=self.scheme.lower() == 'https')
    p.write_conf(extra='test_case_name: "test_4_Autoreload"')
    p.start(imports='cherrypy.test._test_states_demo')
    try:
        self.getPage('/start')
        start = float(self.body)
        time.sleep(2)
        os.utime(os.path.join(thisdir, '_test_states_demo.py'), None)
        time.sleep(2)
        host = cherrypy.server.socket_host
        port = cherrypy.server.socket_port
        portend.occupied(host, port, timeout=5)
        self.getPage('/start')
        if not float(self.body) > start:
            raise AssertionError('start time %s not greater than %s' % (float(self.body), start))
    finally:
        self.getPage('/exit')
    p.join()