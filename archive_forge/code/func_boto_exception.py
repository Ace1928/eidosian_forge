import json
import os
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.ansible_release import __version__
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import binary_type
from ansible.module_utils.six import text_type
from .common import get_collection_info
from .exceptions import AnsibleBotocoreError
from .retries import AWSRetry
def boto_exception(err):
    """
    Extracts the error message from a boto exception.

    :param err: Exception from boto
    :return: Error message
    """
    if hasattr(err, 'error_message'):
        error = err.error_message
    elif hasattr(err, 'message'):
        error = str(err.message) + ' ' + str(err) + ' - ' + str(type(err))
    else:
        error = f'{Exception}: {err}'
    return error