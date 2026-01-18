import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('name', metavar='<name>', type=str, help=_('Name of the module.'))
@utils.arg('type', metavar='<type>', type=str, help=_('Type of the module. The type must be supported by a corresponding module plugin on the datastore it is applied to.'))
@utils.arg('file', metavar='<filename>', type=argparse.FileType(mode='rb', bufsize=0), help=_('File containing data contents for the module.'))
@utils.arg('--description', metavar='<description>', type=str, help=_('Description of the module.'))
@utils.arg('--datastore', metavar='<datastore>', help=_('Name or ID of datastore this module can be applied to. If not specified, module can be applied to all datastores.'))
@utils.arg('--datastore_version', metavar='<version>', help=_('Name or ID of datastore version this module can be applied to. If not specified, module can be applied to all versions.'))
@utils.arg('--auto_apply', action='store_true', default=False, help=_('Automatically apply this module when creating an instance or cluster. Admin only.'))
@utils.arg('--all_tenants', action='store_true', default=False, help=_('Module is valid for all tenants. Admin only.'))
@utils.arg('--hidden', action='store_true', default=False, help=_('Hide this module from non-Admin. Useful in creating auto-apply modules without cluttering up module lists. Admin only.'))
@utils.arg('--live_update', action='store_true', default=False, help=_('Allow module to be updated even if it is already applied to a current instance or cluster.'))
@utils.arg('--priority_apply', action='store_true', default=False, help=_('Sets a priority for applying the module. All priority modules will be applied before non-priority ones. Admin only.'))
@utils.arg('--apply_order', type=int, default=5, choices=range(0, 10), help=_('Sets an order for applying the module. Modules with a lower value will be applied before modules with a higher value. Modules having the same value may be applied in any order (default %(default)s).'))
@utils.arg('--full_access', action='store_true', default=None, help=_("Marks a module as 'non-admin', unless an admin-only option was specified. Admin only."))
@utils.service_type('database')
def do_module_create(cs, args):
    """Create a module."""
    contents = args.file.read()
    if not contents:
        raise exceptions.ValidationError(_("The file '%s' must contain some data") % args.file)
    module = cs.modules.create(args.name, args.type, contents, description=args.description, all_tenants=args.all_tenants, datastore=args.datastore, datastore_version=args.datastore_version, auto_apply=args.auto_apply, visible=not args.hidden, live_update=args.live_update, priority_apply=args.priority_apply, apply_order=args.apply_order, full_access=args.full_access)
    _print_object(module)