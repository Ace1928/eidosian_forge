from keystone.common.resource_options import core as ro_core
from keystone.common.validation import parameter_types
from keystone import exception
def check_immutable_delete(resource_ref, resource_type, resource_id):
    """Check if a delete is allowed on a resource.

    :param resource_ref: dict reference of the resource
    :param resource_type: resource type (str) e.g. 'project'
    :param resource_id: id of the resource (str) e.g. project['id']
    :raises: ResourceDeleteForbidden
    """
    if check_resource_immutable(resource_ref):
        raise exception.ResourceDeleteForbidden(type=resource_type, resource_id=resource_id)