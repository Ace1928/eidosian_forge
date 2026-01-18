from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def stub_default_types():
    return {'default_types': [{'project_id': '629632e7-99d2-4c40-9ae3-106fa3b1c9b7', 'volume_type_id': '4c298f16-e339-4c80-b934-6cbfcb7525a0'}, {'project_id': 'a0c01994-1245-416e-8fc9-1aca86329bfd', 'volume_type_id': 'ff094b46-f82a-4a74-9d9e-d3d08116ad93'}]}