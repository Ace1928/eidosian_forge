from __future__ import absolute_import, division, print_function
from ray import Language
from ray._raylet import CppFunctionDescriptor, JavaFunctionDescriptor
from ray.util.annotations import PublicAPI
@PublicAPI(stability='beta')
def cpp_function(function_name: str):
    """Define a Cpp function.

    Args:
        function_name: Cpp function name.
    """
    from ray.remote_function import RemoteFunction
    return RemoteFunction(Language.CPP, lambda *args, **kwargs: None, CppFunctionDescriptor(function_name, 'PYTHON'), {})