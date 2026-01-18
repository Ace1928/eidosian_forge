import logging
import xml.etree.ElementTree as ET
from fiona.env import require_gdal_version
from fiona.ogrext import _get_metadata_item
@require_gdal_version('2.0')
def supported_field_types(driver):
    """ Returns supported field types

    Parameters
    ----------
    driver : str

    Returns
    -------
    list
        List with supported field types or None if not specified by driver

    """
    field_types_str = _get_metadata_item(driver, MetadataItem.CREATION_FIELD_DATA_TYPES)
    if field_types_str is None:
        return None
    return [field_type for field_type in field_types_str.split(' ') if len(field_type) > 0]