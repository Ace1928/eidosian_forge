import io, logging, socket, os, pickle, struct, time, re
from stat import ST_DEV, ST_INO, ST_MTIME
import queue
import threading
import copy
def makePickle(self, record):
    """
        Pickles the record in binary format with a length prefix, and
        returns it ready for transmission across the socket.
        """
    ei = record.exc_info
    if ei:
        dummy = self.format(record)
    d = dict(record.__dict__)
    d['msg'] = record.getMessage()
    d['args'] = None
    d['exc_info'] = None
    d.pop('message', None)
    s = pickle.dumps(d, 1)
    slen = struct.pack('>L', len(s))
    return slen + s