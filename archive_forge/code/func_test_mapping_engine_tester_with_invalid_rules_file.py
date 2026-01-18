import copy
import datetime
import logging
import os
from unittest import mock
import uuid
import argparse
import configparser
import fixtures
import freezegun
import http.client
import oslo_config.fixture
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_upgradecheck import upgradecheck
from testtools import matchers
from keystone.cmd import cli
from keystone.cmd.doctor import caching
from keystone.cmd.doctor import credential
from keystone.cmd.doctor import database as doc_database
from keystone.cmd.doctor import debug
from keystone.cmd.doctor import federation
from keystone.cmd.doctor import ldap
from keystone.cmd.doctor import security_compliance
from keystone.cmd.doctor import tokens
from keystone.cmd.doctor import tokens_fernet
from keystone.cmd import status
from keystone.common import provider_api
from keystone.common.sql import upgrades
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.identity.mapping_backends import mapping as identity_mapping
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.ksfixtures import ldapdb
from keystone.tests.unit.ksfixtures import policy
from keystone.tests.unit.ksfixtures import temporaryfile
from keystone.tests.unit import mapping_fixtures
def test_mapping_engine_tester_with_invalid_rules_file(self):
    tempfilejson = self.useFixture(temporaryfile.SecureTempFile())
    tmpinvalidfile = tempfilejson.file_name
    with open(tmpinvalidfile, 'w') as f:
        f.write('This is an invalid data')
    self.command_rules = tmpinvalidfile
    self.command_input = tmpinvalidfile
    self.command_prefix = None
    self.command_engine_debug = True
    self.useFixture(fixtures.MockPatchObject(CONF, 'command', self.FakeConfCommand(self)))
    mapping_engine = cli.MappingEngineTester()
    self.assertRaises(SystemExit, mapping_engine.main)