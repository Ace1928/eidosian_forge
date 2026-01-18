import logging
import os.path
def get_supported_extensions():
    """Return the list of file extensions for which we have registered compressors."""
    return sorted(_COMPRESSOR_REGISTRY.keys())