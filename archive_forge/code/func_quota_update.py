import sys
import time
import prettytable
from cinderclient import exceptions
from cinderclient import utils
def quota_update(manager, identifier, args):
    updates = {}
    for resource in _quota_resources:
        val = getattr(args, resource, None)
        if val is not None:
            if args.volume_type:
                resource = resource + '_%s' % args.volume_type
            updates[resource] = val
    if updates:
        skip_validation = getattr(args, 'skip_validation', True)
        if not skip_validation:
            updates['skip_validation'] = skip_validation
        quota_show(manager.update(identifier, **updates))
    else:
        msg = 'Must supply at least one quota field to update.'
        raise exceptions.ClientException(code=1, message=msg)