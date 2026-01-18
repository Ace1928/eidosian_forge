from datetime import datetime
from cinderclient.tests.unit import fakes
from cinderclient.tests.unit.v3 import fakes_base
from cinderclient.v3 import client
def get_resource_filters(self, **kw):
    return (200, {}, {'resource_filters': []})