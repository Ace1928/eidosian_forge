import weakref, ctypes, logging, os, glob
from OpenGL.platform import ctypesloader
from OpenGL import _opaque
def open_device(path):
    """Open a particular gbm device
    
    * path -- integer index of devices in sorted enumeration, or
              device basename (`renderD128`) or a full path-name
              as returned from enumerate_devices

    Will raise (at least):

    * RuntimeError for invalid indices
    * IOError/OSError for device access failures
    * RuntimeError if we cannot create the gbm device

    Caller is responsible for calling close_device(display) on the 
    resulting opaque pointer in order to release the open file-handle
    and deallocate the gbm_device record.

    returns GBMDevice, an opaque pointer
    """
    if isinstance(path, int):
        try:
            devices = enumerate_devices()
            path = devices[int]
        except IndexError:
            raise RuntimeError('Only %s devices available, cannot use 0-index %s' % (len(devices), path))
    else:
        path = os.path.join('/dev/dri', path)
    log.debug('Final device path: %s', path)
    fh = open(path, 'w')
    dev = gbm.gbm_create_device(fh.fileno())
    if dev == 0:
        fh.close()
        raise RuntimeError('Unable to create rendering device for: %s' % path)
    _DEVICE_HANDLES[dev] = fh
    return dev