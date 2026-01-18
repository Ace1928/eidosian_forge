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
def store_add_to_backend(image_id, data, size, store, context=None, verifier=None):
    """
    A wrapper around a call to each stores add() method.  This gives glance
    a common place to check the output

    :param image_id:  The image add to which data is added
    :param data: The data to be stored
    :param size: The length of the data in bytes
    :param store: The store to which the data is being added
    :param context: The request context
    :param verifier: An object used to verify signatures for images
    :return: The url location of the file,
             the size amount of data,
             the checksum of the data
             the storage systems metadata dictionary for the location
    """
    location, size, checksum, metadata = store.add(image_id, data, size, context=context, verifier=verifier)
    if metadata is not None:
        _check_metadata(store, metadata)
    return (location, size, checksum, metadata)