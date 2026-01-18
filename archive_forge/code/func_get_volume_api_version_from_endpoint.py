from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def get_volume_api_version_from_endpoint(self):
    magic_tuple = urlparse.urlsplit(self.management_url)
    scheme, netloc, path, query, frag = magic_tuple
    return path.lstrip('/').split('/')[0][1:]