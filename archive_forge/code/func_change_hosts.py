import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def change_hosts(self, mode, host_family, host, onerror=None):
    """mode is either X.HostInsert or X.HostDelete. host_family is
        one of X.FamilyInternet, X.FamilyDECnet or X.FamilyChaos.

        host is a list of bytes. For the Internet family, it should be the
        four bytes of an IPv4 address."""
    request.ChangeHosts(display=self.display, onerror=onerror, mode=mode, host_family=host_family, host=host)