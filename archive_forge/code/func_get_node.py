from .. import utils
from .._lazyload import h5py
from .._lazyload import tables
from decorator import decorator
@with_HDF5
def get_node(f, node):
    """Get a subnode from a HDF5 file or group.

    Parameters
    ----------
    f : tables.File, h5py.File, tables.Group or h5py.Group
        Open HDF5 file handle or node
    node : str
        Name of subnode to retrieve

    Returns
    -------
    g : tables.Group, h5py.Group, tables.CArray or hdf5.Dataset
        Requested HDF5 node.
    """
    if _is_h5py(f, allow_dataset=False):
        return f[node]
    elif _is_tables(f, allow_dataset=False):
        if isinstance(f, tables.File):
            return f.get_node(f.root, node)
        else:
            return f[node]
    else:
        raise TypeError('Expected h5py.File, tables.File, h5py.Group or tables.Group. Got {}'.format(type(f)))