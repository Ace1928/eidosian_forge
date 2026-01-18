import gzip
import io
from unittest import mock
from http.client import IncompleteRead
from urllib.parse import quote as url_quote
import cherrypy
from cherrypy._cpcompat import ntob, ntou
from cherrypy.test import helper
def test_multipart_decoding_no_charset(self):
    body = ntob('\r\n'.join(['--X', 'Content-Disposition: form-data; name="text"', '', 'â\x80\x9c', '--X', 'Content-Disposition: form-data; name="submit"', '', 'Create', '--X--']))
    (self.getPage('/reqparams', method='POST', headers=[('Content-Type', 'multipart/form-data;boundary=X'), ('Content-Length', str(len(body)))], body=body),)
    self.assertBody(b'submit: Create, text: \xe2\x80\x9c')