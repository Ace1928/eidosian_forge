from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def sendCommand(self, server, url, response, follow_redirects=1, secure=0, keyfile=None, certfile=None):
    """posts the protocol buffer to the desired url on the server
    and puts the return data into the protocol buffer 'response'

    NOTE: The underlying socket raises the 'error' exception
    for all I/O related errors (can't connect, etc.).

    If 'response' is None, the server's PB response will be ignored.

    The optional 'follow_redirects' argument indicates the number
    of HTTP redirects that are followed before giving up and raising an
    exception.  The default is 1.

    If 'secure' is true, HTTPS will be used instead of HTTP.  Also,
    'keyfile' and 'certfile' may be set for client authentication.
    """
    data = self.Encode()
    if secure:
        if keyfile and certfile:
            conn = six.moves.http_client.HTTPSConnection(server, key_file=keyfile, cert_file=certfile)
        else:
            conn = six.moves.http_client.HTTPSConnection(server)
    else:
        conn = six.moves.http_client.HTTPConnection(server)
    conn.putrequest('POST', url)
    conn.putheader('Content-Length', '%d' % len(data))
    conn.endheaders()
    conn.send(data)
    resp = conn.getresponse()
    if follow_redirects > 0 and resp.status == 302:
        m = URL_RE.match(resp.getheader('Location'))
        if m:
            protocol, server, url = m.groups()
            return self.sendCommand(server, url, response, follow_redirects=follow_redirects - 1, secure=protocol == 'https', keyfile=keyfile, certfile=certfile)
    if resp.status != 200:
        raise ProtocolBufferReturnError(resp.status)
    if response is not None:
        response.ParseFromString(resp.read())
    return response