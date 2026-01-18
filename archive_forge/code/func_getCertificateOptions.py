import warnings
from typing import TYPE_CHECKING, Any, List, Optional
from OpenSSL import SSL
from twisted.internet._sslverify import _setAcceptableProtocols
from twisted.internet.ssl import (
from twisted.web.client import BrowserLikePolicyForHTTPS
from twisted.web.iweb import IPolicyForHTTPS
from zope.interface.declarations import implementer
from zope.interface.verify import verifyObject
from scrapy.core.downloader.tls import (
from scrapy.settings import BaseSettings
from scrapy.utils.misc import create_instance, load_object
def getCertificateOptions(self) -> CertificateOptions:
    return CertificateOptions(verify=False, method=getattr(self, 'method', getattr(self, '_ssl_method', None)), fixBrokenPeers=True, acceptableCiphers=self.tls_ciphers)