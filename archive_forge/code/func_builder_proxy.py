from functools import wraps
from ..names import XSD_NAMESPACE, XSD_ANY_TYPE
from ..validators import XMLSchema10, XMLSchema11, XsdGroup, \
@wraps(builder)
def builder_proxy(*args, **kwargs):
    obj = builder(*args, **kwargs)
    assert isinstance(obj, XsdComponent)
    if not cls.is_dummy_component(obj):
        cls.components.append(obj)
    else:
        cls.dummy_components.append(obj)
    return obj