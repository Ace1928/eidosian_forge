import socket
from oslo_log import log as logging
from oslo_serialization import jsonutils
from heat.api.aws import exception
from heat.api.aws import utils as api_utils
from heat.common import exception as heat_exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import policy
from heat.common import template_format
from heat.common import urlfetch
from heat.common import wsgi
from heat.rpc import api as rpc_api
from heat.rpc import client as rpc_client
def format_stack_outputs(o):
    keymap = {rpc_api.OUTPUT_DESCRIPTION: 'Description', rpc_api.OUTPUT_KEY: 'OutputKey', rpc_api.OUTPUT_VALUE: 'OutputValue'}

    def replacecolon(d):
        return dict(map(lambda k_v: (k_v[0].replace(':', '.'), k_v[1]), d.items()))

    def transform(attrs):
        """Recursively replace all `:` with `.` in dict keys.

                After that they are not interpreted as xml namespaces.
                """
        new = replacecolon(attrs)
        for key, value in new.items():
            if isinstance(value, dict):
                new[key] = transform(value)
        return new
    return api_utils.reformat_dict_keys(keymap, transform(o))