from __future__ import absolute_import, division, print_function
from ray import Language
from ray._raylet import CppFunctionDescriptor, JavaFunctionDescriptor
from ray.util.annotations import PublicAPI
@PublicAPI(stability='beta')
def java_actor_class(class_name: str):
    """Define a Java actor class.

    Args:
        class_name: Java class name.
    """
    from ray.actor import ActorClass
    return ActorClass._ray_from_function_descriptor(Language.JAVA, JavaFunctionDescriptor(class_name, '<init>', ''), {})