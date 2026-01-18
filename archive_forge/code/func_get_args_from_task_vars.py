from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError
from ansible.plugins.action import ActionBase
from ansible.module_utils.common.arg_spec import ArgumentSpecValidator
from ansible.utils.vars import combine_vars
def get_args_from_task_vars(self, argument_spec, task_vars):
    """
        Get any arguments that may come from `task_vars`.

        Expand templated variables so we can validate the actual values.

        :param argument_spec: A dict of the argument spec.
        :param task_vars: A dict of task variables.

        :returns: A dict of values that can be validated against the arg spec.
        """
    args = {}
    for argument_name, argument_attrs in argument_spec.items():
        if argument_name in task_vars:
            args[argument_name] = task_vars[argument_name]
    args = self._templar.template(args)
    return args