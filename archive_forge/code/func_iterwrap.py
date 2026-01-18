import time
from paste.wsgilib import catch_errors
def iterwrap():
    for chunk in riter:
        environ[ENVIRON_RECEIVED] += len(chunk)
        yield chunk