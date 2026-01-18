from botocore import xform_name
from botocore.model import OperationModel
from botocore.utils import get_service_module_name
from botocore.docs.method import document_model_driven_method
from botocore.docs.method import document_custom_method
from boto3.docs.base import BaseDocumenter
from boto3.docs.method import document_model_driven_resource_method
from boto3.docs.utils import get_resource_ignore_params
from boto3.docs.utils import get_resource_public_actions
from boto3.docs.utils import add_resource_type_overview
def document_action(section, resource_name, event_emitter, action_model, service_model, include_signature=True):
    """Documents a resource action

    :param section: The section to write to

    :param resource_name: The name of the resource

    :param event_emitter: The event emitter to use to emit events

    :param action_model: The model of the action

    :param service_model: The model of the service

    :param include_signature: Whether or not to include the signature.
        It is useful for generating docstrings.
    """
    operation_model = service_model.operation_model(action_model.request.operation)
    ignore_params = get_resource_ignore_params(action_model.request.params)
    example_return_value = 'response'
    if action_model.resource:
        example_return_value = xform_name(action_model.resource.type)
    example_resource_name = xform_name(resource_name)
    if service_model.service_name == resource_name:
        example_resource_name = resource_name
    example_prefix = '%s = %s.%s' % (example_return_value, example_resource_name, action_model.name)
    document_model_driven_resource_method(section=section, method_name=action_model.name, operation_model=operation_model, event_emitter=event_emitter, method_description=operation_model.documentation, example_prefix=example_prefix, exclude_input=ignore_params, resource_action_model=action_model, include_signature=include_signature)