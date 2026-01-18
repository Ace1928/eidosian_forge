from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_text
def guarded_run(runner, module, server=None, generate_additional_results=None):
    suffix = ' for {0}'.format(server) if server is not None else ''
    kwargs = {}
    try:
        return runner()
    except ResolverError as e:
        if generate_additional_results is not None:
            kwargs = generate_additional_results()
        module.fail_json(msg='Unexpected resolving error{0}: {1}'.format(suffix, to_native(e)), exception=traceback.format_exc(), **kwargs)
    except dns.exception.DNSException as e:
        if generate_additional_results is not None:
            kwargs = generate_additional_results()
        module.fail_json(msg='Unexpected DNS error{0}: {1}'.format(suffix, to_native(e)), exception=traceback.format_exc(), **kwargs)