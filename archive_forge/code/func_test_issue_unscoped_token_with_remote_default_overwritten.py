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
def test_issue_unscoped_token_with_remote_default_overwritten(self):
    """Test that protocol remote_id_attribute has higher priority.

        Make sure the parameter stored under ``protocol`` section has higher
        priority over parameter from default ``federation`` configuration
        section.

        """
    self.config_fixture.config(group='saml2', remote_id_attribute=self.REMOTE_ID_ATTR)
    self.config_fixture.config(group='federation', remote_id_attribute=uuid.uuid4().hex)
    self._issue_unscoped_token(idp=self.IDP_WITH_REMOTE, environment={self.REMOTE_ID_ATTR: self.REMOTE_IDS[0]})