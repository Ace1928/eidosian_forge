import argparse
from unittest import mock
from openstack import exceptions
from openstack.identity.v3 import project
import testtools
from osc_lib.cli import identity as cli_identity
from osc_lib.tests import utils as test_utils
def test_add_project_owner_option_to_parser(self):
    parser = argparse.ArgumentParser()
    cli_identity.add_project_owner_option_to_parser(parser)
    parsed_args = parser.parse_args(['--project', 'project1', '--project-domain', 'domain1'])
    self.assertEqual('project1', parsed_args.project)
    self.assertEqual('domain1', parsed_args.project_domain)