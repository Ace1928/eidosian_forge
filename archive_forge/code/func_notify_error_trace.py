from urllib import parse as parser
from oslo_config import cfg
from osprofiler.drivers import base
from osprofiler import exc
def notify_error_trace(self, info):
    """Store base_id and timestamp of error trace to a separate index."""
    self.client.index(index=self.index_name_error, doc_type=self.conf.profiler.es_doc_type, body={'base_id': info['base_id'], 'timestamp': info['timestamp']})