from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_text
from ansible.module_utils.connection import Connection, ConnectionError
from ansible.module_utils.six import iteritems
from ansible.module_utils.six.moves.urllib.parse import urlencode
def parse_splunk_args(module):
    """
    Get the valid fields that should be passed to the REST API as urlencoded
    data so long as the argument specification to the module follows the
    convention:
        1) name field is Required to be passed as data to REST API
        2) all module argspec items that should be passed to data are not
            Required by the module and are set to default=None
    """
    try:
        splunk_data = {}
        for argspec in module.argument_spec:
            if 'default' in module.argument_spec[argspec] and module.argument_spec[argspec]['default'] is None and (module.params[argspec] is not None):
                splunk_data[argspec] = module.params[argspec]
        return splunk_data
    except TypeError as e:
        module.fail_json(msg='Invalid data type provided for splunk module_util.parse_splunk_args: {0}'.format(e))