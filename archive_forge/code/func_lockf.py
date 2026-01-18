import errno
import msvcrt
import pywintypes
import win32con
import win32file
def lockf(fd, flags, length=4294901760, start=0, whence=0):
    overlapped = pywintypes.OVERLAPPED()
    hfile = msvcrt.get_osfhandle(fd.fileno())
    if LOCK_UN & flags:
        ret = win32file.UnlockFileEx(hfile, 0, start, length, overlapped)
    else:
        try:
            ret = win32file.LockFileEx(hfile, flags, start, length, overlapped)
        except:
            raise IOError(errno.EAGAIN, '', '')
    return ret