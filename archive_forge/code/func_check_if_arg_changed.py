from __future__ import absolute_import, division, print_function
import json
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.six.moves.urllib import parse as urllib_parse
from ansible_collections.community.rabbitmq.plugins.module_utils.rabbitmq import rabbitmq_argument_spec
def check_if_arg_changed(module, current_args, desired_args, arg_name):
    if arg_name not in current_args:
        if arg_name in desired_args:
            module.fail_json(msg="RabbitMQ RESTAPI doesn't support attribute changes for existing queues.Attempting to set %s which is not currently set." % arg_name)
    elif arg_name in desired_args:
        if current_args[arg_name] != desired_args[arg_name]:
            module.fail_json(msg="RabbitMQ RESTAPI doesn't support attribute changes for existing queues.\nAttempting to change %s from '%s' to '%s'" % (arg_name, current_args[arg_name], desired_args[arg_name]))
    else:
        module.fail_json(msg="RabbitMQ RESTAPI doesn't support attribute changes for existing queues.Attempting to unset %s which is currently set to '%s'." % (arg_name, current_args[arg_name]))