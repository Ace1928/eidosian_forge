import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('module', metavar='<module>', help=_('ID or name of the module.'))
@utils.service_type('database')
def do_module_delete(cs, args):
    """Delete a module."""
    module = _find_module(cs, args.module)
    cs.modules.delete(module)