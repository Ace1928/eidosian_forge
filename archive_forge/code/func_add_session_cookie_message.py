from django.conf import settings
from .. import Tags, Warning, register
def add_session_cookie_message(message):
    return message + ' Using a secure-only session cookie makes it more difficult for network traffic sniffers to hijack user sessions.'