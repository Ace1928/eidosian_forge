import inspect
import jmespath
from botocore.compat import six
def get_resource_public_actions(resource_class):
    resource_class_members = inspect.getmembers(resource_class)
    resource_methods = {}
    for name, member in resource_class_members:
        if not name.startswith('_'):
            if not name[0].isupper():
                if not name.startswith('wait_until'):
                    if is_resource_action(member):
                        resource_methods[name] = member
    return resource_methods