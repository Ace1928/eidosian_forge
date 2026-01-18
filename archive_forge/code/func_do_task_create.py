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
@utils.arg('--type', metavar='<TYPE>', help=_('Type of Task. Please refer to Glance schema or documentation to see which tasks are supported.'))
@utils.arg('--input', metavar='<STRING>', default='{}', help=_('Parameters of the task to be launched'))
def do_task_create(gc, args):
    """Create a new task."""
    if not (args.type and args.input):
        utils.exit('Unable to create task. Specify task type and input.')
    else:
        try:
            input = json.loads(args.input)
        except ValueError:
            utils.exit('Failed to parse the "input" parameter. Must be a valid JSON object.')
        task_values = {'type': args.type, 'input': input}
        task = gc.tasks.create(**task_values)
        ignore = ['self', 'schema']
        task = dict([item for item in task.items() if item[0] not in ignore])
        utils.print_dict(task)