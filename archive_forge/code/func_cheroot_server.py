from contextlib import closing, contextmanager
import errno
import socket
import threading
import time
import http.client
import pytest
import cheroot.server
from cheroot.test import webtest
import cheroot.wsgi
@contextmanager
def cheroot_server(server_factory):
    """Set up and tear down a Cheroot server instance."""
    conf = config[server_factory].copy()
    bind_port = conf.pop('bind_addr')[-1]
    for interface in (ANY_INTERFACE_IPV6, ANY_INTERFACE_IPV4):
        try:
            actual_bind_addr = (interface, bind_port)
            httpserver = server_factory(bind_addr=actual_bind_addr, **conf)
        except OSError:
            pass
        else:
            break
    httpserver.shutdown_timeout = 0
    threading.Thread(target=httpserver.safe_start).start()
    while not httpserver.ready:
        time.sleep(0.1)
    yield httpserver
    httpserver.stop()