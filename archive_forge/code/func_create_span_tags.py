import collections
import datetime
import time
from urllib import parse as parser
from oslo_config import cfg
from oslo_serialization import jsonutils
from osprofiler import _utils as utils
from osprofiler.drivers import base
from osprofiler import exc
def create_span_tags(self, payload):
    """Create tags for OpenTracing span.

        :param info: Information from OSProfiler trace.
        :returns tags: A dictionary contains standard tags
                       from OpenTracing sematic conventions,
                       and some other custom tags related to http, db calls.
        """
    tags = {}
    info = payload['info']
    if info.get('db'):
        tags['db.statement'] = info['db']['statement']
        tags['db.params'] = jsonutils.dumps(info['db']['params'])
    elif info.get('request'):
        tags['http.path'] = info['request']['path']
        tags['http.query'] = info['request']['query']
        tags['http.method'] = info['request']['method']
        tags['http.scheme'] = info['request']['scheme']
    elif info.get('function'):
        if 'args' in info['function']:
            tags['args'] = info['function']['args']
        if 'kwargs' in info['function']:
            tags['kwargs'] = info['function']['kwargs']
        tags['name'] = info['function']['name']
    return tags