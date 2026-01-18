import os
import signal
import time
from http.client import BadStatusLine
import pytest
import portend
import cherrypy
import cherrypy.process.servers
from cherrypy.test import helper
def test_2_KeyboardInterrupt(self):
    engine.start()
    cherrypy.server.start()
    self.persistent = True
    try:
        self.getPage('/')
        self.assertStatus('200 OK')
        self.assertBody('Hello World')
        self.assertNoHeader('Connection')
        cherrypy.server.httpserver.interrupt = KeyboardInterrupt
        engine.block()
        self.assertEqual(db_connection.running, False)
        self.assertEqual(len(db_connection.threads), 0)
        self.assertEqual(engine.state, engine.states.EXITING)
    finally:
        self.persistent = False
    engine.start()
    cherrypy.server.start()
    try:
        self.getPage('/ctrlc', raise_subcls=BadStatusLine)
    except BadStatusLine:
        pass
    else:
        print(self.body)
        self.fail('AssertionError: BadStatusLine not raised')
    engine.block()
    self.assertEqual(db_connection.running, False)
    self.assertEqual(len(db_connection.threads), 0)