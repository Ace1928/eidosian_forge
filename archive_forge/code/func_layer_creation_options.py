import logging
import xml.etree.ElementTree as ET
from fiona.env import require_gdal_version
from fiona.ogrext import _get_metadata_item
@require_gdal_version('2.0')
def layer_creation_options(driver):
    """ Returns layer creation options for driver

    Parameters
    ----------
    driver : str

    Returns
    -------
    dict
        Layer creation options

    """
    xml = _get_metadata_item(driver, MetadataItem.LAYER_CREATION_OPTION_LIST)
    if xml is None:
        return {}
    if len(xml) == 0:
        return {}
    return _parse_options(xml)