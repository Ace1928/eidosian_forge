import abc
from neutron_lib._i18n import _
from neutron_lib import constants
def get_request_extensions(self):
    """List of extensions.RequestExtension extension objects.

        Request extensions are used to handle custom request data.
        """
    return []