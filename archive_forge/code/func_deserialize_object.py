import pickle
import warnings
from itertools import chain
from jupyter_client.session import MAX_BYTES, MAX_ITEMS
def deserialize_object(buffers, g=None):
    """reconstruct an object serialized by serialize_object from data buffers.

    Parameters
    ----------
    buffers : list of buffers/bytes
    g : globals to be used when uncanning

    Returns
    -------
    (newobj, bufs) : unpacked object, and the list of remaining unused buffers.
    """
    bufs = list(buffers)
    pobj = bufs.pop(0)
    canned = pickle.loads(pobj)
    if istype(canned, sequence_types) and len(canned) < MAX_ITEMS:
        for c in canned:
            _restore_buffers(c, bufs)
        newobj = uncan_sequence(canned, g)
    elif istype(canned, dict) and len(canned) < MAX_ITEMS:
        newobj = {}
        for k in sorted(canned):
            c = canned[k]
            _restore_buffers(c, bufs)
            newobj[k] = uncan(c, g)
    else:
        _restore_buffers(canned, bufs)
        newobj = uncan(canned, g)
    return (newobj, bufs)