import errno
import sys
import pickle
def pickle_dump(data, file, protocol):
    try:
        pickle.dump(data, file, protocol)
        file.flush()
    except OSError:
        if sys.platform == 'win32':
            raise IOError(errno.EPIPE, 'Broken pipe')
        raise