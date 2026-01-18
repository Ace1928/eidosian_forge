from datetime import datetime
from cinderclient.tests.unit import fakes
from cinderclient.tests.unit.v3 import fakes_base
from cinderclient.v3 import client
def get_clusters(self, **kw):
    clusters = self._filter_clusters(self.CLUSTER_SUMMARY_KEYS, **kw)
    return (200, {}, {'clusters': clusters})