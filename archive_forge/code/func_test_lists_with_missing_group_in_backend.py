import copy
import os
import random
import re
import subprocess
from testtools import matchers
from unittest import mock
import uuid
import fixtures
import flask
import http.client
from lxml import etree
from oslo_serialization import jsonutils
from oslo_utils import importutils
import saml2
from saml2 import saml
from saml2 import sigver
import urllib
from keystone.api._shared import authentication
from keystone.api import auth as auth_api
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common import render_token
import keystone.conf
from keystone import exception
from keystone.federation import idp as keystone_idp
from keystone.models import token_model
from keystone import notifications
from keystone.tests import unit
from keystone.tests.unit import core
from keystone.tests.unit import federation_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
def test_lists_with_missing_group_in_backend(self):
    """Test a mapping that points to a group that does not exist.

        For explicit mappings, we expect the group to exist in the backend,
        but for lists, specifically blacklists, a missing group is expected
        as many groups will be specified by the IdP that are not Keystone
        groups.

        The test scenario is as follows:
         - Create group ``EXISTS``
         - Set mapping rules for existing IdP with a blacklist
           that passes through as REMOTE_USER_GROUPS
         - Issue unscoped token with on group  ``EXISTS`` id in it

        """
    domain_id = self.domainA['id']
    domain_name = self.domainA['name']
    group = unit.new_group_ref(domain_id=domain_id, name='EXISTS')
    group = PROVIDERS.identity_api.create_group(group)
    rules = {'rules': [{'local': [{'user': {'name': '{0}', 'id': '{0}'}}], 'remote': [{'type': 'REMOTE_USER'}]}, {'local': [{'groups': '{0}', 'domain': {'name': domain_name}}], 'remote': [{'type': 'REMOTE_USER_GROUPS'}]}]}
    PROVIDERS.federation_api.update_mapping(self.mapping['id'], rules)
    r = self._issue_unscoped_token(assertion='UNMATCHED_GROUP_ASSERTION')
    assigned_group_ids = r.federated_groups
    self.assertEqual(1, len(assigned_group_ids))
    self.assertEqual(group['id'], assigned_group_ids[0]['id'])