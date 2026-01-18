import flask
import uuid
from oslo_config import fixture as config_fixture
from oslo_serialization import jsonutils
from keystone.auth.plugins import mapped
import keystone.conf
from keystone import exception
from keystone.federation import utils as mapping_utils
from keystone.tests import unit
from keystone.tests.unit import mapping_fixtures
from unittest import mock
def test_mapping_validation_with_user_name_and_domain_name(self):
    mapping = mapping_fixtures.MAPPING_WITH_USERNAME_AND_DOMAINNAME
    mapping_utils.validate_mapping_structure(mapping)