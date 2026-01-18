import argparse
from http import server
import socketserver
import webob
from oslo_middleware import healthcheck
@webob.dec.wsgify
def dummy_application(req):
    return 'test'