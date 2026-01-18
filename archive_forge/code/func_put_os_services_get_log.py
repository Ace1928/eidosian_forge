from datetime import datetime
from cinderclient.tests.unit import fakes
from cinderclient.tests.unit.v3 import fakes_base
from cinderclient.v3 import client
def put_os_services_get_log(self, body):
    levels = [{'binary': 'cinder-api', 'host': 'host1', 'levels': {'prefix1': 'DEBUG', 'prefix2': 'INFO'}}, {'binary': 'cinder-volume', 'host': 'host@backend#pool', 'levels': {'prefix3': 'WARNING', 'prefix4': 'ERROR'}}]
    return (200, {}, {'log_levels': levels})