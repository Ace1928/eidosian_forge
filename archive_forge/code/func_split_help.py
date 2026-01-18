import abc
import contextlib
import logging
import openstack.exceptions
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from openstackclient.i18n import _
from openstackclient.network import utils
@staticmethod
def split_help(network_help, compute_help):
    return '*%(network_qualifier)s:*\n  %(network_help)s\n\n*%(compute_qualifier)s:*\n  %(compute_help)s' % dict(network_qualifier=_('Network version 2'), network_help=network_help, compute_qualifier=_('Compute version 2'), compute_help=compute_help)