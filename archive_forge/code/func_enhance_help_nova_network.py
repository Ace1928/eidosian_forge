import abc
import contextlib
import logging
import openstack.exceptions
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from openstackclient.i18n import _
from openstackclient.network import utils
def enhance_help_nova_network(self, _help):
    if self.is_docs_build:
        return _QUALIFIER_FMT % (_help, _('Compute version 2 only'))
    return _help