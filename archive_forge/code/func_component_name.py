from ._base import *
def component_name(name: str, module: str=None):
    """OpenAPI components must be unique by name"""

    def decorator(obj):
        obj.__name__ = name
        obj.__qualname__ = name
        if module is not None:
            obj.__module__ = module
        key = (obj.__name__, obj.__module__)
        if key in components:
            if components[key].schema() != obj.schema():
                raise RuntimeError(f'Different models with the same name detected: {obj!r} != {components[key]}')
            return components[key]
        components[key] = obj
        return obj
    return decorator