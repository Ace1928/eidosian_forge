import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('configuration_group', metavar='<configuration_group>', help=_('ID or name of the configuration group.'))
@utils.arg('values', metavar='<values>', help=_('Dictionary of the values to set.'))
@utils.arg('--name', metavar='<name>', default=None, help=_('Name of the configuration group.'))
@utils.arg('--description', metavar='<description>', default=None, help=_('An optional description for the configuration group.'))
@utils.service_type('database')
def do_configuration_update(cs, args):
    """Updates a configuration group."""
    configuration = _find_configuration(cs, args.configuration_group)
    cs.configurations.update(configuration, args.values, args.name, args.description)