from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def convert_group_ids_sourced_from_list(index_list, reference_data):
    value_list = []
    for group_index in index_list:
        value_list.append(reference_data['groups'][group_index]['id'])
    return value_list