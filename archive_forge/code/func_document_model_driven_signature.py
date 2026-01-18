import inspect
import types
from botocore.docs.example import (
from botocore.docs.params import (
def document_model_driven_signature(section, name, operation_model, include=None, exclude=None):
    """Documents the signature of a model-driven method

    :param section: The section to write the documentation to.

    :param name: The name of the method

    :param operation_model: The operation model for the method

    :type include: Dictionary where keys are parameter names and
        values are the shapes of the parameter names.
    :param include: The parameter shapes to include in the documentation.

    :type exclude: List of the names of the parameters to exclude.
    :param exclude: The names of the parameters to exclude from
        documentation.
    """
    params = {}
    if operation_model.input_shape:
        params = operation_model.input_shape.members
    parameter_names = list(params.keys())
    if include is not None:
        for member in include:
            parameter_names.append(member.name)
    if exclude is not None:
        for member in exclude:
            if member in parameter_names:
                parameter_names.remove(member)
    signature_params = ''
    if parameter_names:
        signature_params = '**kwargs'
    section.style.start_sphinx_py_method(name, signature_params)