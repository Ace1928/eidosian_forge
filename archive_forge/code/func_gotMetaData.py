from twisted import copyright
from twisted.web import http
def gotMetaData(self, metadata):
    """Called with a list of (key, value) pairs of metadata,
        if metadata is available on the server.

        Will only be called on non-empty metadata.
        """
    raise NotImplementedError('implement in subclass')