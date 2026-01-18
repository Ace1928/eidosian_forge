import inspect
import types
from botocore.docs.example import (
from botocore.docs.params import (
def document_model_driven_method(section, method_name, operation_model, event_emitter, method_description=None, example_prefix=None, include_input=None, include_output=None, exclude_input=None, exclude_output=None, document_output=True, include_signature=True):
    """Documents an individual method

    :param section: The section to write to

    :param method_name: The name of the method

    :param operation_model: The model of the operation

    :param event_emitter: The event emitter to use to emit events

    :param example_prefix: The prefix to use in the method example.

    :type include_input: Dictionary where keys are parameter names and
        values are the shapes of the parameter names.
    :param include_input: The parameter shapes to include in the
        input documentation.

    :type include_output: Dictionary where keys are parameter names and
        values are the shapes of the parameter names.
    :param include_input: The parameter shapes to include in the
        output documentation.

    :type exclude_input: List of the names of the parameters to exclude.
    :param exclude_input: The names of the parameters to exclude from
        input documentation.

    :type exclude_output: List of the names of the parameters to exclude.
    :param exclude_input: The names of the parameters to exclude from
        output documentation.

    :param document_output: A boolean flag to indicate whether to
        document the output.

    :param include_signature: Whether or not to include the signature.
        It is useful for generating docstrings.
    """
    if include_signature:
        document_model_driven_signature(section, method_name, operation_model, include=include_input, exclude=exclude_input)
    method_intro_section = section.add_new_section('method-intro')
    method_intro_section.include_doc_string(method_description)
    if operation_model.deprecated:
        method_intro_section.style.start_danger()
        method_intro_section.writeln('This operation is deprecated and may not function as expected. This operation should not be used going forward and is only kept for the purpose of backwards compatiblity.')
        method_intro_section.style.end_danger()
    service_uid = operation_model.service_model.metadata.get('uid')
    if service_uid is not None:
        method_intro_section.style.new_paragraph()
        method_intro_section.write('See also: ')
        link = f'{AWS_DOC_BASE}/{service_uid}/{operation_model.name}'
        method_intro_section.style.external_link(title='AWS API Documentation', link=link)
        method_intro_section.writeln('')
    example_section = section.add_new_section('request-example')
    example_section.style.new_paragraph()
    example_section.style.bold('Request Syntax')
    context = {'special_shape_types': {'streaming_input_shape': operation_model.get_streaming_input(), 'streaming_output_shape': operation_model.get_streaming_output(), 'eventstream_output_shape': operation_model.get_event_stream_output()}}
    if operation_model.input_shape:
        RequestExampleDocumenter(service_name=operation_model.service_model.service_name, operation_name=operation_model.name, event_emitter=event_emitter, context=context).document_example(example_section, operation_model.input_shape, prefix=example_prefix, include=include_input, exclude=exclude_input)
    else:
        example_section.style.new_paragraph()
        example_section.style.start_codeblock()
        example_section.write(example_prefix + '()')
    request_params_section = section.add_new_section('request-params')
    if operation_model.input_shape:
        RequestParamsDocumenter(service_name=operation_model.service_model.service_name, operation_name=operation_model.name, event_emitter=event_emitter, context=context).document_params(request_params_section, operation_model.input_shape, include=include_input, exclude=exclude_input)
    return_section = section.add_new_section('return')
    return_section.style.new_line()
    if operation_model.output_shape is not None and document_output:
        return_section.write(':rtype: dict')
        return_section.style.new_line()
        return_section.write(':returns: ')
        return_section.style.indent()
        return_section.style.new_line()
        event_stream_output = operation_model.get_event_stream_output()
        if event_stream_output:
            event_section = return_section.add_new_section('event-stream')
            event_section.style.new_paragraph()
            event_section.write('The response of this operation contains an :class:`.EventStream` member. When iterated the :class:`.EventStream` will yield events based on the structure below, where only one of the top level keys will be present for any given event.')
            event_section.style.new_line()
        return_example_section = return_section.add_new_section('response-example')
        return_example_section.style.new_line()
        return_example_section.style.bold('Response Syntax')
        return_example_section.style.new_paragraph()
        ResponseExampleDocumenter(service_name=operation_model.service_model.service_name, operation_name=operation_model.name, event_emitter=event_emitter, context=context).document_example(return_example_section, operation_model.output_shape, include=include_output, exclude=exclude_output)
        return_description_section = return_section.add_new_section('description')
        return_description_section.style.new_line()
        return_description_section.style.bold('Response Structure')
        return_description_section.style.new_paragraph()
        ResponseParamsDocumenter(service_name=operation_model.service_model.service_name, operation_name=operation_model.name, event_emitter=event_emitter, context=context).document_params(return_description_section, operation_model.output_shape, include=include_output, exclude=exclude_output)
    else:
        return_section.write(':returns: None')