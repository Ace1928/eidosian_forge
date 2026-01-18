from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible.module_utils.six.moves import shlex_quote
from ansible.module_utils.basic import AnsibleModule
def sanitize_output(output):
    """
    Sanitize the output string before we
    pass it to module.fail_json. Defaults
    the string to empty if it is None, else
    strips trailing newlines.

    :param output: output to sanitize
    :return: sanitized output
    """
    if output is None:
        return ''
    else:
        return output.rstrip('\r\n')