import getpass
import io
import urllib.parse, urllib.request
from warnings import warn
from distutils.core import PyPIRCCommand
from distutils.errors import *
from distutils import log
def classifiers(self):
    """ Fetch the list of classifiers from the server.
        """
    url = self.repository + '?:action=list_classifiers'
    response = urllib.request.urlopen(url)
    log.info(self._read_pypi_response(response))