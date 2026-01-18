import warnings
import json
from tarfile import TarFile
from pkgutil import get_data
from io import BytesIO
from dateutil.tz import tzfile as _tzfile
def gettz(name):
    """
    This retrieves a time zone from the local zoneinfo tarball that is packaged
    with dateutil.

    :param name:
        An IANA-style time zone name, as found in the zoneinfo file.

    :return:
        Returns a :class:`dateutil.tz.tzfile` time zone object.

    .. warning::
        It is generally inadvisable to use this function, and it is only
        provided for API compatibility with earlier versions. This is *not*
        equivalent to ``dateutil.tz.gettz()``, which selects an appropriate
        time zone based on the inputs, favoring system zoneinfo. This is ONLY
        for accessing the dateutil-specific zoneinfo (which may be out of
        date compared to the system zoneinfo).

    .. deprecated:: 2.6
        If you need to use a specific zoneinfofile over the system zoneinfo,
        instantiate a :class:`dateutil.zoneinfo.ZoneInfoFile` object and call
        :func:`dateutil.zoneinfo.ZoneInfoFile.get(name)` instead.

        Use :func:`get_zonefile_instance` to retrieve an instance of the
        dateutil-provided zoneinfo.
    """
    warnings.warn('zoneinfo.gettz() will be removed in future versions, to use the dateutil-provided zoneinfo files, instantiate a ZoneInfoFile object and use ZoneInfoFile.zones.get() instead. See the documentation for details.', DeprecationWarning)
    if len(_CLASS_ZONE_INSTANCE) == 0:
        _CLASS_ZONE_INSTANCE.append(ZoneInfoFile(getzoneinfofile_stream()))
    return _CLASS_ZONE_INSTANCE[0].zones.get(name)