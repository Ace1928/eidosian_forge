from typing import Optional, Union
from urllib.parse import urlparse
import dns.asyncbackend
import dns.asyncquery
import dns.inet
import dns.message
import dns.query
def answer_port(self) -> int:
    port = urlparse(self.url).port
    if port is None:
        port = 443
    return port