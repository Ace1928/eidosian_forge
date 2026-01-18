from packaging.version import Version
from requests.adapters import HTTPAdapter
from docker.transport.basehttpadapter import BaseHTTPAdapter
import urllib3
def can_override_ssl_version(self):
    urllib_ver = urllib3.__version__.split('-')[0]
    if urllib_ver is None:
        return False
    if urllib_ver == 'dev':
        return True
    return Version(urllib_ver) > Version('1.5')