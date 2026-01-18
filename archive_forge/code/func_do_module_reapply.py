import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('module', metavar='<module>', type=str, help=_('Name or ID of the module.'))
@utils.arg('--md5', metavar='<md5>', type=str, default=None, help=_('Reapply the module only to instances applied with the specific md5.'))
@utils.arg('--include_clustered', action='store_true', default=False, help=_('Include instances that are part of a cluster (default %(default)s).'))
@utils.arg('--batch_size', metavar='<batch_size>', type=int, default=None, help=_('Number of instances to reapply the module to before sleeping.'))
@utils.arg('--delay', metavar='<delay>', type=int, default=None, help=_('Time to sleep in seconds between applying batches.'))
@utils.arg('--force', action='store_true', default=False, help=_('Force reapply even on modules already having the current MD5'))
@utils.service_type('database')
def do_module_reapply(cs, args):
    """Reapply a module."""
    module = _find_module(cs, args.module)
    cs.modules.reapply(module, args.md5, args.include_clustered, args.batch_size, args.delay, args.force)