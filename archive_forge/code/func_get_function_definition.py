import inspect
from yaql.language import exceptions
from yaql.language import utils
from yaql.language import yaqltypes
def get_function_definition(func, name=None, function=None, method=None, convention=None, parameter_type_func=None):
    if parameter_type_func is None:
        parameter_type_func = _infer_parameter_type
    fd = _get_function_definition(func).clone()
    spec = inspect.getfullargspec(func)
    for arg in spec.args + spec.kwonlyargs:
        if arg not in fd.parameters:
            fd.set_parameter(arg, parameter_type_func(arg))
    if spec.varargs and '*' not in fd.parameters:
        fd.set_parameter(spec.varargs, parameter_type_func(spec.varargs))
    if spec.varkw and '**' not in fd.parameters:
        fd.set_parameter(spec.varkw, parameter_type_func(spec.varkw))
    if name is not None:
        fd.name = name
    elif fd.name is None:
        fd.name = convert_function_name(fd.payload.__name__, convention)
    elif convention is not None:
        fd.name = convert_function_name(fd.name, convention)
    if function is not None:
        fd.is_function = function
    if method is not None:
        fd.is_method = method
    if convention:
        for p in fd.parameters.values():
            if p.alias is None:
                p.alias = convert_parameter_name(p.name, convention)
    return fd