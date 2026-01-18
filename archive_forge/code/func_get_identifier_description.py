import inspect
import jmespath
from botocore.compat import six
def get_identifier_description(resource_name, identifier_name):
    return "The %s's %s identifier. This **must** be set." % (resource_name, identifier_name)