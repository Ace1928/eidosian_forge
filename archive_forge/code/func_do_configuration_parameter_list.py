import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('--datastore', metavar='<datastore>', default=None, help=_('ID or name of the datastore to list configuration parameters for. Optional if the ID of the datastore_version is provided.'))
@utils.arg('datastore_version', metavar='<datastore_version>', help=_('Datastore version name or ID assigned to the configuration group.'))
@utils.service_type('database')
def do_configuration_parameter_list(cs, args):
    """Lists available parameters for a configuration group."""
    if args.datastore:
        params = cs.configuration_parameters.parameters(args.datastore, args.datastore_version)
    elif utils.is_uuid_like(args.datastore_version):
        params = cs.configuration_parameters.parameters_by_version(args.datastore_version)
    else:
        raise exceptions.NoUniqueMatch(_('The datastore name or id is required to retrieve the parameters for the configuration group by name.'))
    for param in params:
        setattr(param, 'min', getattr(param, 'min', '-'))
        setattr(param, 'max', getattr(param, 'max', '-'))
    utils.print_list(params, ['name', 'type', 'min', 'max', 'restart_required'], labels={'min': 'Min Size', 'max': 'Max Size'})