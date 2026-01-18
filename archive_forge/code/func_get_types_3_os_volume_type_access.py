from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def get_types_3_os_volume_type_access(self, **kw):
    return (200, {}, {'volume_type_access': [_stub_type_access()]})