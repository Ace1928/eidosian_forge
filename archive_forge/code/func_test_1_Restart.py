import os
import signal
import time
from http.client import BadStatusLine
import pytest
import portend
import cherrypy
import cherrypy.process.servers
from cherrypy.test import helper
def test_1_Restart(self):
    cherrypy.server.start()
    engine.start()
    self.assertEqual(db_connection.running, True)
    grace = db_connection.gracecount
    self.getPage('/')
    self.assertBody('Hello World')
    self.assertEqual(len(db_connection.threads), 1)
    engine.graceful()
    self.assertEqual(engine.state, engine.states.STARTED)
    self.getPage('/')
    self.assertBody('Hello World')
    self.assertEqual(db_connection.running, True)
    self.assertEqual(db_connection.gracecount, grace + 1)
    self.assertEqual(len(db_connection.threads), 1)
    self.getPage('/graceful')
    self.assertEqual(engine.state, engine.states.STARTED)
    self.assertBody('app was (gracefully) restarted succesfully')
    self.assertEqual(db_connection.running, True)
    self.assertEqual(db_connection.gracecount, grace + 2)
    self.assertEqual(len(db_connection.threads), 0)
    engine.stop()
    self.assertEqual(engine.state, engine.states.STOPPED)
    self.assertEqual(db_connection.running, False)
    self.assertEqual(len(db_connection.threads), 0)