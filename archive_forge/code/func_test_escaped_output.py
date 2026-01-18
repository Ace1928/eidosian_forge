import datetime
import logging
from cheroot.test import webtest
import pytest
import requests  # FIXME: Temporary using it directly, better switch
import cherrypy
from cherrypy.test.logtest import LogCase
def test_escaped_output(log_tracker, server):
    log_tracker.markLog()
    host = webtest.interface(webtest.WebCase.HOST)
    port = webtest.WebCase.PORT
    resp = requests.get('http://%s:%s/uni_code' % (host, port))
    assert resp.status_code == 200
    log_tracker.assertLog(-1, repr(tartaros.encode('utf8'))[2:-1])
    log_tracker.assertLog(-1, '\\xce\\x88\\xcf\\x81\\xce\\xb5\\xce\\xb2\\xce\\xbf\\xcf\\x82')
    log_tracker.markLog()
    resp = requests.get('http://%s:%s/slashes' % (host, port))
    assert resp.status_code == 200
    log_tracker.assertLog(-1, b'"GET /slashed\\path HTTP/1.1"')
    log_tracker.markLog()
    resp = requests.get('http://%s:%s/whitespace' % (host, port))
    assert resp.status_code == 200
    log_tracker.assertLog(-1, '"Browzuh (1.0\\r\\n\\t\\t.3)"')