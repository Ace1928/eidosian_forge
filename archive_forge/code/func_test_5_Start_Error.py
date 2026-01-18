import os
import signal
import time
from http.client import BadStatusLine
import pytest
import portend
import cherrypy
import cherrypy.process.servers
from cherrypy.test import helper
def test_5_Start_Error(self):
    if engine.state != engine.states.EXITING:
        engine.exit()
    p = helper.CPProcess(ssl=self.scheme.lower() == 'https', wait=True)
    p.write_conf(extra='starterror: True\ntest_case_name: "test_5_Start_Error"\n')
    p.start(imports='cherrypy.test._test_states_demo')
    if p.exit_code == 0:
        self.fail('Process failed to return nonzero exit code.')