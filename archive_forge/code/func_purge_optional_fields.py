from __future__ import absolute_import, division, print_function
import os
import re
import logging
import sys
from copy import deepcopy
from ansible.module_utils.basic import env_fallback
def purge_optional_fields(obj, module):
    """
    It purges the optional arguments to be sent to the controller.
    :param obj: dictionary of the ansible object passed as argument.
    :param module: AnsibleModule
    return modified obj
    """
    purge_fields = []
    for param, spec in module.argument_spec.items():
        if not spec.get('required', False):
            if param not in obj:
                continue
            if obj[param] is None:
                purge_fields.append(param)
    log.debug('purging fields %s', purge_fields)
    for param in purge_fields:
        obj.pop(param, None)
    return obj