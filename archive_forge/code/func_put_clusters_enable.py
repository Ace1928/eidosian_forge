from datetime import datetime
from cinderclient.tests.unit import fakes
from cinderclient.tests.unit.v3 import fakes_base
from cinderclient.v3 import client
def put_clusters_enable(self, body):
    res = self.get_clusters(id=1)
    return (200, {}, {'cluster': res[2]['clusters'][0]})