from .. import utils
from .._lazyload import h5py
from .._lazyload import tables
from decorator import decorator
@with_HDF5
def list_nodes(f):
    """List all first-level nodes in a HDF5 file.

    Parameters
    ----------
    f : tables.File or h5py.File
        Open HDF5 file handle.

    Returns
    -------
    nodes : list
        List of names of first-level nodes below f
    """
    if _is_h5py(f, allow_dataset=False):
        return [node for node in f.keys()]
    elif _is_tables(f, allow_dataset=False):
        return [node._v_name for node in f.list_nodes(f.root)]
    else:
        raise TypeError('Expected h5py.File, tables.File, h5py.Group or tables.Group. Got {}'.format(type(f)))