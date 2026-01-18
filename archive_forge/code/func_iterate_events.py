from urllib import parse as parser
from debtcollector import removals
from oslo_config import cfg
from oslo_serialization import jsonutils
from osprofiler.drivers import base
from osprofiler import exc
def iterate_events():
    for key in self.db.scan_iter(match=self.namespace + base_id + '*'):
        yield self.db.get(key)
    for event in self.db.lrange(self.namespace_opt + base_id, 0, -1):
        yield event