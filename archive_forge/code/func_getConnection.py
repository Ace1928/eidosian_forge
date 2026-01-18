import io, logging, socket, os, pickle, struct, time, re
from stat import ST_DEV, ST_INO, ST_MTIME
import queue
import threading
import copy
def getConnection(self, host, secure):
    """
        get a HTTP[S]Connection.

        Override when a custom connection is required, for example if
        there is a proxy.
        """
    import http.client
    if secure:
        connection = http.client.HTTPSConnection(host, context=self.context)
    else:
        connection = http.client.HTTPConnection(host)
    return connection