import gzip
import io
from unittest import mock
from http.client import IncompleteRead
from urllib.parse import quote as url_quote
import cherrypy
from cherrypy._cpcompat import ntob, ntou
from cherrypy.test import helper
@mock.patch('cherrypy._cpreqbody.Part.maxrambytes', 1)
def test_multipart_decoding_bigger_maxrambytes(self):
    """
        Decoding of a multipart entity should also pass when
        the entity is bigger than maxrambytes. See ticket #1352.
        """
    self.test_multipart_decoding()