import json
import os
import sys
from oslo_utils import strutils
from glanceclient._i18n import _
from glanceclient.common import progressbar
from glanceclient.common import utils
from glanceclient import exc
from glanceclient.v2 import cache
from glanceclient.v2 import image_members
from glanceclient.v2 import image_schema
from glanceclient.v2 import images
from glanceclient.v2 import namespace_schema
from glanceclient.v2 import resource_type_schema
from glanceclient.v2 import tasks
@utils.arg('--sort-key', default='status', choices=tasks.SORT_KEY_VALUES, help=_('Sort task list by specified field.'))
@utils.arg('--sort-dir', default='desc', choices=tasks.SORT_DIR_VALUES, help=_('Sort task list in specified direction.'))
@utils.arg('--page-size', metavar='<SIZE>', default=None, type=int, help=_('Number of tasks to request in each paginated request.'))
@utils.arg('--type', metavar='<TYPE>', help=_('Filter tasks to those that have this type.'))
@utils.arg('--status', metavar='<STATUS>', help=_('Filter tasks to those that have this status.'))
def do_task_list(gc, args):
    """List tasks you can access."""
    filter_keys = ['type', 'status']
    filter_items = [(key, getattr(args, key)) for key in filter_keys]
    filters = dict([item for item in filter_items if item[1] is not None])
    kwargs = {'filters': filters}
    if args.page_size is not None:
        kwargs['page_size'] = args.page_size
    kwargs['sort_key'] = args.sort_key
    kwargs['sort_dir'] = args.sort_dir
    tasks = gc.tasks.list(**kwargs)
    columns = ['ID', 'Type', 'Status', 'Owner']
    utils.print_list(tasks, columns)