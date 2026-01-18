import warnings
import json
from tarfile import TarFile
from pkgutil import get_data
from io import BytesIO
from dateutil.tz import tzfile as _tzfile
def getzoneinfofile_stream():
    try:
        return BytesIO(get_data(__name__, ZONEFILENAME))
    except IOError as e:
        warnings.warn('I/O error({0}): {1}'.format(e.errno, e.strerror))
        return None