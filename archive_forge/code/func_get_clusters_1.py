from datetime import datetime
from cinderclient.tests.unit import fakes
from cinderclient.tests.unit.v3 import fakes_base
from cinderclient.v3 import client
def get_clusters_1(self):
    res = self.get_clusters_detail(id=1)
    return (200, {}, {'cluster': res[2]['clusters'][0]})