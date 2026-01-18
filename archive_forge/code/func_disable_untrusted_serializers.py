from kombu.serialization import disable_insecure_serializers as _disable_insecure_serializers
from kombu.serialization import registry
from celery.exceptions import ImproperlyConfigured
from .serialization import register_auth  # : need cryptography first
def disable_untrusted_serializers(whitelist=None):
    _disable_insecure_serializers(allowed=whitelist)