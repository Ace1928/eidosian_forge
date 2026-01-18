from . import exceptions
from . import misc
from . import normalizers
def ensure_components_are_valid(uri, validated_components):
    """Assert that all components are valid in the URI."""
    invalid_components = set([])
    for component in validated_components:
        if component in _SUBAUTHORITY_VALIDATORS:
            if not subauthority_component_is_valid(uri, component):
                invalid_components.add(component)
            continue
        validator = _COMPONENT_VALIDATORS[component]
        if not validator(getattr(uri, component)):
            invalid_components.add(component)
    if invalid_components:
        raise exceptions.InvalidComponentsError(uri, *invalid_components)