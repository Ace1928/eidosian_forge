import threading
import socket
import time
import pytest
import requests
from tests.testserver.server import Server
def test_basic_response(self):
    """the basic response server returns an empty http response"""
    with Server.basic_response_server() as (host, port):
        r = requests.get('http://{}:{}'.format(host, port))
        assert r.status_code == 200
        assert r.text == u''
        assert r.headers['Content-Length'] == '0'