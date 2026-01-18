from __future__ import absolute_import, division, print_function
from ray import Language
from ray._raylet import CppFunctionDescriptor, JavaFunctionDescriptor
from ray.util.annotations import PublicAPI
@PublicAPI(stability='beta')
def java_function(class_name: str, function_name: str):
    """Define a Java function.

    Args:
        class_name: Java class name.
        function_name: Java function name.
    """
    from ray.remote_function import RemoteFunction
    return RemoteFunction(Language.JAVA, lambda *args, **kwargs: None, JavaFunctionDescriptor(class_name, function_name, ''), {})