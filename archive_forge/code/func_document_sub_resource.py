from botocore import xform_name
from botocore.utils import get_service_module_name
from boto3.docs.base import BaseDocumenter
from boto3.docs.utils import get_identifier_args_for_signature
from boto3.docs.utils import get_identifier_values_for_example
from boto3.docs.utils import get_identifier_description
from boto3.docs.utils import add_resource_type_overview
def document_sub_resource(section, resource_name, sub_resource_model, service_model, include_signature=True):
    """Documents a resource action

    :param section: The section to write to

    :param resource_name: The name of the resource

    :param sub_resource_model: The model of the subresource

    :param service_model: The model of the service

    :param include_signature: Whether or not to include the signature.
        It is useful for generating docstrings.
    """
    identifiers_needed = []
    for identifier in sub_resource_model.resource.identifiers:
        if identifier.source == 'input':
            identifiers_needed.append(xform_name(identifier.target))
    if include_signature:
        signature_args = get_identifier_args_for_signature(identifiers_needed)
        section.style.start_sphinx_py_method(sub_resource_model.name, signature_args)
    method_intro_section = section.add_new_section('method-intro')
    description = 'Creates a %s resource.' % sub_resource_model.resource.type
    method_intro_section.include_doc_string(description)
    example_section = section.add_new_section('example')
    example_values = get_identifier_values_for_example(identifiers_needed)
    example_resource_name = xform_name(resource_name)
    if service_model.service_name == resource_name:
        example_resource_name = resource_name
    example = '%s = %s.%s(%s)' % (xform_name(sub_resource_model.resource.type), example_resource_name, sub_resource_model.name, example_values)
    example_section.style.start_codeblock()
    example_section.write(example)
    example_section.style.end_codeblock()
    param_section = section.add_new_section('params')
    for identifier in identifiers_needed:
        description = get_identifier_description(sub_resource_model.name, identifier)
        param_section.write(':type %s: string' % identifier)
        param_section.style.new_line()
        param_section.write(':param %s: %s' % (identifier, description))
        param_section.style.new_line()
    return_section = section.add_new_section('return')
    return_section.style.new_line()
    return_section.write(':rtype: :py:class:`%s.%s`' % (get_service_module_name(service_model), sub_resource_model.resource.type))
    return_section.style.new_line()
    return_section.write(':returns: A %s resource' % sub_resource_model.resource.type)
    return_section.style.new_line()