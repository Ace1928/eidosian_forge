import hashlib
import logging
from oslo_config import cfg
from oslo_utils import encodeutils
from stevedore import driver
from stevedore import extension
from glance_store import capabilities
from glance_store import exceptions
from glance_store.i18n import _
from glance_store import location
def store_add_to_backend_with_multihash(image_id, data, size, hashing_algo, store, context=None, verifier=None):
    """
    A wrapper around a call to each store's add() method that requires
    a hashing_algo identifier and returns a 5-tuple including the
    "multihash" computed using the specified hashing_algo.  (This
    is an enhanced version of store_add_to_backend(), which is left
    as-is for backward compatibility.)

    :param image_id:  The image add to which data is added
    :param data: The data to be stored
    :param size: The length of the data in bytes
    :param store: The store to which the data is being added
    :param hashing_algo: A hashlib algorithm identifier (string)
    :param context: The request context
    :param verifier: An object used to verify signatures for images
    :return: The url location of the file,
             the size amount of data,
             the checksum of the data,
             the multihash of the data,
             the storage system's metadata dictionary for the location
    :raises: ``glance_store.exceptions.BackendException``
             ``glance_store.exceptions.UnknownHashingAlgo``
    """
    if hashing_algo not in hashlib.algorithms_available:
        raise exceptions.UnknownHashingAlgo(algo=hashing_algo)
    location, size, checksum, multihash, metadata = store.add(image_id, data, size, hashing_algo, context=context, verifier=verifier)
    if metadata is not None:
        _check_metadata(store, metadata)
    return (location, size, checksum, multihash, metadata)