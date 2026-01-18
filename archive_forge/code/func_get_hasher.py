import hashlib
import logging
import uuid
from oslo_concurrency import lockutils
from oslo_utils.secretutils import md5
from glance_store.i18n import _
def get_hasher(hash_algo, usedforsecurity=True):
    """
    Returns the required hasher, given the hashing algorithm.
    This is primarily to ensure that the hash algorithm is correctly
    chosen when executed on a FIPS enabled system

    :param hash_algo: hash algorithm requested
    :param usedforsecurity: whether the hashes are used in a security context
    """
    if str(hash_algo) == 'md5':
        return md5(usedforsecurity=usedforsecurity)
    else:
        return hashlib.new(str(hash_algo))