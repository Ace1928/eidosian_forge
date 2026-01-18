import os
import time
from twisted.internet import defer
from twisted.names import common, dns, error
from twisted.python import failure
from twisted.python.compat import execfile, nativeString
from twisted.python.filepath import FilePath
def getSerial(filename='/tmp/twisted-names.serial'):
    """
    Return a monotonically increasing (across program runs) integer.

    State is stored in the given file.  If it does not exist, it is
    created with rw-/---/--- permissions.

    This manipulates process-global state by calling C{os.umask()}, so it isn't
    thread-safe.

    @param filename: Path to a file that is used to store the state across
        program runs.
    @type filename: L{str}

    @return: a monotonically increasing number
    @rtype: L{str}
    """
    serial = time.strftime('%Y%m%d')
    o = os.umask(127)
    try:
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                f.write(serial + ' 0')
    finally:
        os.umask(o)
    with open(filename) as serialFile:
        lastSerial, zoneID = serialFile.readline().split()
    zoneID = lastSerial == serial and int(zoneID) + 1 or 0
    with open(filename, 'w') as serialFile:
        serialFile.write('%s %d' % (serial, zoneID))
    serial = serial + '%02d' % (zoneID,)
    return serial