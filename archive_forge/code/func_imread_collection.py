import skimage.io as io
def imread_collection(load_pattern, conserve_memory=True):
    """Load a collection of images from one or more FITS files

    Parameters
    ----------
    load_pattern : str or list
        List of extensions to load. Filename globbing is currently
        unsupported.
    conserve_memory : bool
        If True, never keep more than one in memory at a specific
        time. Otherwise, images will be cached once they are loaded.

    Returns
    -------
    ic : ImageCollection
        Collection of images.

    """
    intype = type(load_pattern)
    if intype is not list and intype is not str:
        raise TypeError('Input must be a filename or list of filenames')
    if intype is not list:
        load_pattern = [load_pattern]
    ext_list = []
    for filename in load_pattern:
        with fits.open(filename) as hdulist:
            for n, hdu in zip(range(len(hdulist)), hdulist):
                if isinstance(hdu, fits.ImageHDU) or isinstance(hdu, fits.PrimaryHDU):
                    try:
                        data_size = hdu.size
                    except TypeError:
                        data_size = hdu.size()
                    if data_size > 0:
                        ext_list.append((filename, n))
    return io.ImageCollection(ext_list, load_func=FITSFactory, conserve_memory=conserve_memory)