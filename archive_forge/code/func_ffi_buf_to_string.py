from typing import Any, Optional
import OpenSSL._util as pyOpenSSLutil
import OpenSSL.SSL
import OpenSSL.version
from OpenSSL.crypto import X509Name
from scrapy.utils.python import to_unicode
def ffi_buf_to_string(buf: Any) -> str:
    return to_unicode(pyOpenSSLutil.ffi.string(buf))