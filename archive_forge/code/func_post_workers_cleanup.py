from datetime import datetime
from cinderclient.tests.unit import fakes
from cinderclient.tests.unit.v3 import fakes_base
from cinderclient.v3 import client
def post_workers_cleanup(self, **kw):
    response = {'cleaning': [{'id': '1', 'cluster_name': 'cluster1', 'host': 'host1', 'binary': 'binary'}, {'id': '3', 'cluster_name': 'cluster1', 'host': 'host3', 'binary': 'binary'}], 'unavailable': [{'id': '2', 'cluster_name': 'cluster2', 'host': 'host2', 'binary': 'binary'}]}
    return (200, {}, response)