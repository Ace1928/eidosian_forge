import logging
import urllib.parse
from oslo_config import cfg
from glance_store import exceptions
from glance_store.i18n import _
def get_store_uri(self):
    """
        Returns the Glance image URI, which is the host:port of the API server
        along with /images/<IMAGE_ID>
        """
    return self.store_location.get_uri()