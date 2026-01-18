import inspect
import jmespath
from botocore.compat import six
def get_resource_ignore_params(params):
    """Helper method to determine which parameters to ignore for actions

    :returns: A list of the parameter names that does not need to be
        included in a resource's method call for documentation purposes.
    """
    ignore_params = []
    for param in params:
        result = jmespath.compile(param.target)
        current = result.parsed
        while current['children']:
            current = current['children'][0]
        if current['type'] == 'field':
            ignore_params.append(current['value'])
    return ignore_params