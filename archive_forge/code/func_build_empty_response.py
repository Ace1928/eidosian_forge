import jmespath
from botocore import xform_name
from .params import get_data_member
def build_empty_response(search_path, operation_name, service_model):
    """
    Creates an appropriate empty response for the type that is expected,
    based on the service model's shape type. For example, a value that
    is normally a list would then return an empty list. A structure would
    return an empty dict, and a number would return None.

    :type search_path: string
    :param search_path: JMESPath expression to search in the response
    :type operation_name: string
    :param operation_name: Name of the underlying service operation.
    :type service_model: :ref:`botocore.model.ServiceModel`
    :param service_model: The Botocore service model
    :rtype: dict, list, or None
    :return: An appropriate empty value
    """
    response = None
    operation_model = service_model.operation_model(operation_name)
    shape = operation_model.output_shape
    if search_path:
        for item in search_path.split('.'):
            item = item.strip('[0123456789]$')
            if shape.type_name == 'structure':
                shape = shape.members[item]
            elif shape.type_name == 'list':
                shape = shape.member
            else:
                raise NotImplementedError('Search path hits shape type {0} from {1}'.format(shape.type_name, item))
    if shape.type_name == 'structure':
        response = {}
    elif shape.type_name == 'list':
        response = []
    elif shape.type_name == 'map':
        response = {}
    return response