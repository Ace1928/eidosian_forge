from boto.compat import json
from tests.compat import mock, unittest
from tests.unit.cloudsearch.test_search import HOSTNAME, \
from boto.cloudsearch.search import SearchConnection, SearchServiceException
def fake_loads_value_error(content, *args, **kwargs):
    """Callable to generate a fake ValueError"""
    raise ValueError('HAHAHA! Totally not simplejson & you gave me bad JSON.')