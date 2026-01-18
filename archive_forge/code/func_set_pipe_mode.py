import sys
def set_pipe_mode(hPipe, mode=-1, maxCollectionCount=None, collectDataTimeout=None):
    if mode == -1:
        mode = win32pipe.PIPE_READMODE_BYTE
    try:
        win32pipe.SetNamedPipeHandleState(hPipe, mode, maxCollectionCount, collectDataTimeout)
    except pywintypes.error:
        raise