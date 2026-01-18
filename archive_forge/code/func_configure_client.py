import os
import ssl
from . import errors
from .transport import SSLHTTPAdapter
def configure_client(self, client):
    """
        Configure a client with these TLS options.
        """
    client.ssl_version = self.ssl_version
    if self.verify and self.ca_cert:
        client.verify = self.ca_cert
    else:
        client.verify = self.verify
    if self.cert:
        client.cert = self.cert
    client.mount('https://', SSLHTTPAdapter(ssl_version=self.ssl_version, assert_hostname=self.assert_hostname, assert_fingerprint=self.assert_fingerprint))