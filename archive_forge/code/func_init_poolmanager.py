from requests.adapters import HTTPAdapter
from .._compat import poolmanager, basestring
def init_poolmanager(self, connections, maxsize, block=False):
    self.poolmanager = poolmanager.PoolManager(num_pools=connections, maxsize=maxsize, block=block, source_address=self.source_address)