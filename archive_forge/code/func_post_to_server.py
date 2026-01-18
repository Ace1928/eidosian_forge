import getpass
import io
import urllib.parse, urllib.request
from warnings import warn
from distutils.core import PyPIRCCommand
from distutils.errors import *
from distutils import log
def post_to_server(self, data, auth=None):
    """ Post a query to the server, and return a string response.
        """
    if 'name' in data:
        self.announce('Registering %s to %s' % (data['name'], self.repository), log.INFO)
    boundary = '--------------GHSKFJDLGDS7543FJKLFHRE75642756743254'
    sep_boundary = '\n--' + boundary
    end_boundary = sep_boundary + '--'
    body = io.StringIO()
    for key, value in data.items():
        if type(value) not in (type([]), type(())):
            value = [value]
        for value in value:
            value = str(value)
            body.write(sep_boundary)
            body.write('\nContent-Disposition: form-data; name="%s"' % key)
            body.write('\n\n')
            body.write(value)
            if value and value[-1] == '\r':
                body.write('\n')
    body.write(end_boundary)
    body.write('\n')
    body = body.getvalue().encode('utf-8')
    headers = {'Content-type': 'multipart/form-data; boundary=%s; charset=utf-8' % boundary, 'Content-length': str(len(body))}
    req = urllib.request.Request(self.repository, body, headers)
    opener = urllib.request.build_opener(urllib.request.HTTPBasicAuthHandler(password_mgr=auth))
    data = ''
    try:
        result = opener.open(req)
    except urllib.error.HTTPError as e:
        if self.show_response:
            data = e.fp.read()
        result = (e.code, e.msg)
    except urllib.error.URLError as e:
        result = (500, str(e))
    else:
        if self.show_response:
            data = self._read_pypi_response(result)
        result = (200, 'OK')
    if self.show_response:
        msg = '\n'.join(('-' * 75, data, '-' * 75))
        self.announce(msg, log.INFO)
    return result