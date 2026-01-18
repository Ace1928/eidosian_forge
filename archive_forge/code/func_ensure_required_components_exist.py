from . import exceptions
from . import misc
from . import normalizers
def ensure_required_components_exist(uri, required_components):
    """Assert that all required components are present in the URI."""
    missing_components = sorted([component for component in required_components if getattr(uri, component) is None])
    if missing_components:
        raise exceptions.MissingComponentError(uri, *missing_components)