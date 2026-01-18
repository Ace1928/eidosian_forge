import six
from pycadf import cadftaxonomy
def convert_req_action(method, details=None):
    """Maps standard HTTP methods to equivalent CADF action

    :param method: HTTP request method
    :param details: Extra details to append to action.
    """
    mapping = {'get': cadftaxonomy.ACTION_READ, 'head': cadftaxonomy.ACTION_READ, 'post': cadftaxonomy.ACTION_CREATE, 'put': cadftaxonomy.ACTION_UPDATE, 'delete': cadftaxonomy.ACTION_DELETE, 'patch': cadftaxonomy.ACTION_UPDATE, 'options': cadftaxonomy.ACTION_READ, 'trace': 'capture'}
    action = None
    if isinstance(method, six.string_types):
        action = mapping.get(method.lower())
        if action and isinstance(details, six.string_types):
            action += '/%s' % details
    return action or cadftaxonomy.UNKNOWN