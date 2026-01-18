import json
import six
from os_service_types import data
from os_service_types.tests import base
def create_json(self, json_data):
    fd, name = self.create_temp_file(suffix='.json')
    with fd:
        json.dump(json_data, fd)
    return name