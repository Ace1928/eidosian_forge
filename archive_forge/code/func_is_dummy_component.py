from functools import wraps
from ..names import XSD_NAMESPACE, XSD_ANY_TYPE
from ..validators import XMLSchema10, XMLSchema11, XsdGroup, \
@classmethod
def is_dummy_component(cls, component) -> bool:
    if component.parent in cls.dummy_components:
        return True
    elif isinstance(component, XsdAttributeGroup):
        return not component
    elif isinstance(component, XsdComplexType):
        return component.name == XSD_ANY_TYPE and component.target_namespace != XSD_NAMESPACE
    elif isinstance(component, XsdGroup) and component.parent is not None:
        return component.parent.name == XSD_ANY_TYPE and component.target_namespace != XSD_NAMESPACE
    return False