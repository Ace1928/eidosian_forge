from base64 import b64decode
from paste.httpexceptions import HTTPUnauthorized
from paste.httpheaders import (
def build_authentication(self):
    head = WWW_AUTHENTICATE.tuples('Basic realm="%s"' % self.realm)
    return HTTPUnauthorized(headers=head)