import warnings
import json
from tarfile import TarFile
from pkgutil import get_data
from io import BytesIO
from dateutil.tz import tzfile as _tzfile
def get_zonefile_instance(new_instance=False):
    """
    This is a convenience function which provides a :class:`ZoneInfoFile`
    instance using the data provided by the ``dateutil`` package. By default, it
    caches a single instance of the ZoneInfoFile object and returns that.

    :param new_instance:
        If ``True``, a new instance of :class:`ZoneInfoFile` is instantiated and
        used as the cached instance for the next call. Otherwise, new instances
        are created only as necessary.

    :return:
        Returns a :class:`ZoneInfoFile` object.

    .. versionadded:: 2.6
    """
    if new_instance:
        zif = None
    else:
        zif = getattr(get_zonefile_instance, '_cached_instance', None)
    if zif is None:
        zif = ZoneInfoFile(getzoneinfofile_stream())
        get_zonefile_instance._cached_instance = zif
    return zif