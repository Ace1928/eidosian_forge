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
def paginated_query_with_retries(client, paginator_name, retry_decorator=None, **params):
    """
    Performs a boto3 paginated query.
    By default uses uses AWSRetry.jittered_backoff(retries=10) to retry queries
    with temporary failures.

    Examples:
        tags = paginated_query_with_retries(client, "describe_tags", Filters=[])

        decorator = AWSRetry.backoff(tries=5, delay=5, backoff=2.0,
                                     catch_extra_error_codes=['RequestInProgressException'])
        certificates = paginated_query_with_retries(client, "list_certificates", decorator)
    """
    if retry_decorator is None:
        retry_decorator = AWSRetry.jittered_backoff(retries=10)
    result = retry_decorator(_paginated_query)(client, paginator_name, **params)
    return result