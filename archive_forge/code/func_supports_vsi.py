import logging
import xml.etree.ElementTree as ET
from fiona.env import require_gdal_version
from fiona.ogrext import _get_metadata_item
@require_gdal_version('2.0')
def supports_vsi(driver):
    """ Returns True if driver supports GDAL's VSI*L API

    Parameters
    ----------
    driver : str

    Returns
    -------
    bool

    """
    virutal_io = _get_metadata_item(driver, MetadataItem.VIRTUAL_IO)
    return virutal_io is not None and virutal_io.upper() == 'YES'