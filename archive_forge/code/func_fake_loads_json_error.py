from boto.compat import json
from tests.compat import mock, unittest
from tests.unit.cloudsearch.test_search import HOSTNAME, \
from boto.cloudsearch.search import SearchConnection, SearchServiceException
def fake_loads_json_error(content, *args, **kwargs):
    """Callable to generate a fake JSONDecodeError"""
    raise json.JSONDecodeError('Using simplejson & you gave me bad JSON.', '', 0)