from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def delete_os_volume_transfer_5678(self, **kw):
    return (202, {}, None)