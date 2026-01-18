import io
import sys
from unittest import mock
from oslo_utils import encodeutils
from requests import Response
import testtools
from glanceclient.common import utils
def schema_getter(_type='string', enum=False):
    prop = {'type': ['null', _type], 'description': 'Test schema'}
    prop_readonly = {'type': ['null', _type], 'readOnly': True, 'description': 'Test schema read-only'}
    if enum:
        prop['enum'] = [None, 'opt-1', 'opt-2']
        prop_readonly['enum'] = [None, 'opt-ro-1', 'opt-ro-2']

    def actual_getter():
        return {'additionalProperties': False, 'required': ['name'], 'name': 'test_schema', 'properties': {'test': prop, 'readonly-test': prop_readonly}}
    return actual_getter