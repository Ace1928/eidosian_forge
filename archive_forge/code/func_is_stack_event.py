import collections
import time
from openstack.cloud import meta
from openstack import exceptions
def is_stack_event(event):
    if event.get('resource_name', '') != stack_name and event.get('physical_resource_id', '') != stack_name:
        return False
    phys_id = event.get('physical_resource_id', '')
    links = dict(((link.get('rel'), link.get('href')) for link in event.get('links', [])))
    stack_id = links.get('stack', phys_id).rsplit('/', 1)[-1]
    return stack_id == phys_id