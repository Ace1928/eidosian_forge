from eventlet.zipkin import http
from eventlet.zipkin import wsgi
from eventlet.zipkin import greenthread
from eventlet.zipkin import log
from eventlet.zipkin import api
from eventlet.zipkin.client import ZipkinClient
def disable_trace_patch():
    http.unpatch()
    wsgi.unpatch()
    greenthread.unpatch()
    log.unpatch()
    api.client.close()