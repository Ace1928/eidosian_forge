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
def document_load_reload_action(section, action_name, resource_name, event_emitter, load_model, service_model, include_signature=True):
    """Documents the resource load action

    :param section: The section to write to

    :param action_name: The name of the loading action should be load or reload

    :param resource_name: The name of the resource

    :param event_emitter: The event emitter to use to emit events

    :param load_model: The model of the load action

    :param service_model: The model of the service

    :param include_signature: Whether or not to include the signature.
        It is useful for generating docstrings.
    """
    description = 'Calls  :py:meth:`%s.Client.%s` to update the attributes of the %s resource. Note that the load and reload methods are the same method and can be used interchangeably.' % (get_service_module_name(service_model), xform_name(load_model.request.operation), resource_name)
    example_resource_name = xform_name(resource_name)
    if service_model.service_name == resource_name:
        example_resource_name = resource_name
    example_prefix = '%s.%s' % (example_resource_name, action_name)
    document_model_driven_method(section=section, method_name=action_name, operation_model=OperationModel({}, service_model), event_emitter=event_emitter, method_description=description, example_prefix=example_prefix, include_signature=include_signature)