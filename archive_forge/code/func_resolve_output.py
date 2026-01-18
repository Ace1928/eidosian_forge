import logging
import sys
from oslo_serialization import jsonutils
from oslo_utils import strutils
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient.common import deployment_utils
from heatclient.common import event_utils
from heatclient.common import hook_utils
from heatclient.common import http
from heatclient.common import template_format
from heatclient.common import template_utils
from heatclient.common import utils
import heatclient.exc as exc
def resolve_output(output_key):
    try:
        output = hc.stacks.output_show(args.id, output_key)
    except exc.HTTPNotFound:
        try:
            output = None
            stack = hc.stacks.get(args.id).to_dict()
            for o in stack.get('outputs', []):
                if o['output_key'] == output_key:
                    output = {'output': o}
                    break
            if output is None:
                raise exc.CommandError(_('Output %(key)s not found.') % {'key': args.output})
        except exc.HTTPNotFound:
            raise exc.CommandError(_('Stack %(id)s or output %(key)s not found.') % {'id': args.id, 'key': args.output})
    return output