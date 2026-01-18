from botocore import xform_name
from botocore.utils import get_service_module_name
from botocore.docs.method import document_model_driven_method
from boto3.docs.base import BaseDocumenter
from boto3.docs.utils import get_resource_ignore_params
from boto3.docs.utils import add_resource_type_overview
def document_resource_waiter(section, resource_name, event_emitter, service_model, resource_waiter_model, service_waiter_model, include_signature=True):
    waiter_model = service_waiter_model.get_waiter(resource_waiter_model.waiter_name)
    operation_model = service_model.operation_model(waiter_model.operation)
    ignore_params = get_resource_ignore_params(resource_waiter_model.params)
    service_module_name = get_service_module_name(service_model)
    description = 'Waits until this %s is %s. This method calls :py:meth:`%s.Waiter.%s.wait` which polls. :py:meth:`%s.Client.%s` every %s seconds until a successful state is reached. An error is returned after %s failed checks.' % (resource_name, ' '.join(resource_waiter_model.name.split('_')[2:]), service_module_name, xform_name(resource_waiter_model.waiter_name), service_module_name, xform_name(waiter_model.operation), waiter_model.delay, waiter_model.max_attempts)
    example_prefix = '%s.%s' % (xform_name(resource_name), resource_waiter_model.name)
    document_model_driven_method(section=section, method_name=resource_waiter_model.name, operation_model=operation_model, event_emitter=event_emitter, example_prefix=example_prefix, method_description=description, exclude_input=ignore_params, include_signature=include_signature)
    if 'return' in section.available_sections:
        return_section = section.get_section('return')
        return_section.clear_text()
        return_section.remove_all_sections()
        return_section.write(':returns: None')