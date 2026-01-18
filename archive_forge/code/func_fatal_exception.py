from lazyops.imports._aiohttpx import aiohttpx
from lazyops.types.classprops import lazyproperty
def fatal_exception(exc):
    if isinstance(exc, ClientError):
        return exc.status_code == 503 or exc.status_code >= 400
    else:
        return False