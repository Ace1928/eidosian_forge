from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import text_type
from ansible.plugins.action import ActionBase
from ansible.utils.vars import merge_hash
from ..module_utils import bonsai, errors
@staticmethod
def build_asset_args(args, bonsai_args):
    asset_args = dict(name=args.get('rename', args['name']), state='present', builds=bonsai_args['builds'])
    if 'auth' in args:
        asset_args['auth'] = args['auth']
    if 'namespace' in args:
        asset_args['namespace'] = args['namespace']
    for meta in ('labels', 'annotations'):
        if bonsai_args[meta] or args.get(meta):
            asset_args[meta] = merge_hash(bonsai_args[meta] or {}, args.get(meta, {}))
    return asset_args