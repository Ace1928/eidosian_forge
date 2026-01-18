import logging
import os.path
def compression_wrapper(file_obj, mode, compression=INFER_FROM_EXTENSION, filename=None):
    """
    Wrap `file_obj` with an appropriate [de]compression mechanism based on its file extension.

    If the filename extension isn't recognized, simply return the original `file_obj` unchanged.

    `file_obj` must either be a filehandle object, or a class which behaves like one.

    If `filename` is specified, it will be used to extract the extension.
    If not, the `file_obj.name` attribute is used as the filename.

    """
    if compression == NO_COMPRESSION:
        return file_obj
    elif compression == INFER_FROM_EXTENSION:
        try:
            filename = (filename or file_obj.name).lower()
        except (AttributeError, TypeError):
            logger.warning('unable to transparently decompress %r because it seems to lack a string-like .name', file_obj)
            return file_obj
        _, compression = os.path.splitext(filename)
    if compression in _COMPRESSOR_REGISTRY and mode.endswith('+'):
        raise ValueError('transparent (de)compression unsupported for mode %r' % mode)
    try:
        callback = _COMPRESSOR_REGISTRY[compression]
    except KeyError:
        return file_obj
    else:
        return callback(file_obj, mode)