import six
from google.auth import exceptions
from google.auth.transport import _mtls_helper
def has_default_client_cert_source():
    """Check if default client SSL credentials exists on the device.

    Returns:
        bool: indicating if the default client cert source exists.
    """
    metadata_path = _mtls_helper._check_dca_metadata_path(_mtls_helper.CONTEXT_AWARE_METADATA_PATH)
    return metadata_path is not None