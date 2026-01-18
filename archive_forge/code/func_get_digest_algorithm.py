import sys
from contextlib import contextmanager
import cryptography.exceptions
from cryptography.hazmat.primitives import hashes
from celery.exceptions import SecurityError, reraise
def get_digest_algorithm(digest='sha256'):
    """Convert string to hash object of cryptography library."""
    assert digest is not None
    return getattr(hashes, digest.upper())()