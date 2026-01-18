import struct
from django.core.exceptions import ValidationError
from .const import (
def from_pgraster(data):
    """
    Convert a PostGIS HEX String into a dictionary.
    """
    if data is None:
        return
    header, data = chunk(data, 122)
    header = unpack(POSTGIS_HEADER_STRUCTURE, header)
    bands = []
    pixeltypes = []
    while data:
        pixeltype_with_flags, data = chunk(data, 2)
        pixeltype_with_flags = unpack('B', pixeltype_with_flags)[0]
        pixeltype = pixeltype_with_flags & BANDTYPE_PIXTYPE_MASK
        pixeltype = POSTGIS_TO_GDAL[pixeltype]
        pack_type = GDAL_TO_STRUCT[pixeltype]
        pack_size = 2 * STRUCT_SIZE[pack_type]
        nodata, data = chunk(data, pack_size)
        nodata = unpack(pack_type, nodata)[0]
        band, data = chunk(data, pack_size * header[10] * header[11])
        band_result = {'data': bytes.fromhex(band)}
        if pixeltype_with_flags & BANDTYPE_FLAG_HASNODATA:
            band_result['nodata_value'] = nodata
        bands.append(band_result)
        pixeltypes.append(pixeltype)
    if len(set(pixeltypes)) != 1:
        raise ValidationError('Band pixeltypes are not all equal.')
    return {'srid': int(header[9]), 'width': header[10], 'height': header[11], 'datatype': pixeltypes[0], 'origin': (header[5], header[6]), 'scale': (header[3], header[4]), 'skew': (header[7], header[8]), 'bands': bands}