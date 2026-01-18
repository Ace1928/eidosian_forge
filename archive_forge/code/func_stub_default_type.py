from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def stub_default_type():
    return {'default_type': {'project_id': '629632e7-99d2-4c40-9ae3-106fa3b1c9b7', 'volume_type_id': '4c298f16-e339-4c80-b934-6cbfcb7525a0'}}