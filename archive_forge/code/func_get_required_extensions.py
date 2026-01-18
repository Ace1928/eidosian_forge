import abc
from neutron_lib._i18n import _
from neutron_lib import constants
@classmethod
def get_required_extensions(cls):
    """Returns the API definition's required extensions."""
    cls._assert_api_definition('REQUIRED_EXTENSIONS')
    return cls.api_definition.REQUIRED_EXTENSIONS