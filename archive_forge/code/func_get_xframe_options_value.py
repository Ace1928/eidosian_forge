from django.conf import settings
from django.utils.deprecation import MiddlewareMixin
def get_xframe_options_value(self, request, response):
    """
        Get the value to set for the X_FRAME_OPTIONS header. Use the value from
        the X_FRAME_OPTIONS setting, or 'DENY' if not set.

        This method can be overridden if needed, allowing it to vary based on
        the request or response.
        """
    return getattr(settings, 'X_FRAME_OPTIONS', 'DENY').upper()