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
def test_empty_whitelist_discards_all_values(self):
    """Test that empty whitelist blocks all the values.

        Not adding a ``whitelist`` keyword to the mapping value is different
        than adding empty whitelist.  The former case will simply pass all the
        values, whereas the latter would discard all the values.

        This test checks scenario where an empty whitelist was specified.
        The expected result is that no groups are matched.

        The test scenario is as follows:
         - Create group ``EXISTS``
         - Set mapping rules for existing IdP with an empty whitelist
           that whould discard any values from the assertion
         - Try issuing unscoped token, no groups were matched and that the
           federated user does not have any group assigned.

        """
    domain_id = self.domainA['id']
    domain_name = self.domainA['name']
    group = unit.new_group_ref(domain_id=domain_id, name='EXISTS')
    group = PROVIDERS.identity_api.create_group(group)
    rules = {'rules': [{'local': [{'user': {'name': '{0}', 'id': '{0}'}}], 'remote': [{'type': 'REMOTE_USER'}]}, {'local': [{'groups': '{0}', 'domain': {'name': domain_name}}], 'remote': [{'type': 'REMOTE_USER_GROUPS', 'whitelist': []}]}]}
    PROVIDERS.federation_api.update_mapping(self.mapping['id'], rules)
    r = self._issue_unscoped_token(assertion='UNMATCHED_GROUP_ASSERTION')
    assigned_groups = r.federated_groups
    self.assertEqual(len(assigned_groups), 0)