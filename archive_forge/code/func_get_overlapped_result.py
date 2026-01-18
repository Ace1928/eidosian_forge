import sys
def get_overlapped_result(handle, overlapped=None, bWait=False):
    try:
        return win32file.GetOverlappedResult(handle, overlapped, bWait)
    except pywintypes.error:
        raise