import json
import os
import sys
def query_user(prompt, default=None):
    """Query the user for data.

    Args:
        prompt: (str) Prompt to display to the user.
        default: (str or None) Default value to use if the user doesn't input
            anything.

    Returns:
        (str) Value returned by the user.
    """
    kwargs = {}
    kwargs['prompt'] = prompt
    if default is not None:
        kwargs['default'] = default
    _write_msg(type='query_user', **kwargs)
    return json.loads(sys.stdin.readline())['result']