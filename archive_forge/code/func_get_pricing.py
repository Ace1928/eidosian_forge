import re
import os.path
from typing import Dict, Union, Optional
from os.path import join as pjoin
def get_pricing(driver_type, driver_name, pricing_file_path=None, cache_all=False):
    """
    Return pricing for the provided driver.

    NOTE: This method will also cache data for the requested driver
    memory.

    We intentionally only cache data for the requested driver and not all the
    pricing data since the whole pricing data is quite large (~2 MB). This
    way we avoid unnecessary memory overhead.

    :type driver_type: ``str``
    :param driver_type: Driver type ('compute' or 'storage')

    :type driver_name: ``str``
    :param driver_name: Driver name

    :type pricing_file_path: ``str``
    :param pricing_file_path: Custom path to a price file. If not provided
                              it uses a default path.

    :type cache_all: ``bool``
    :param cache_all: True to cache pricing data in memory for all the drivers
                      and not just for the requested one.

    :rtype: ``dict``
    :return: Dictionary with pricing where a key name is size ID and
             the value is a price.
    """
    cache_all = cache_all or CACHE_ALL_PRICING_DATA
    if driver_type not in VALID_PRICING_DRIVER_TYPES:
        raise AttributeError('Invalid driver type: %s', driver_type)
    if driver_name in PRICING_DATA[driver_type]:
        return PRICING_DATA[driver_type][driver_name]
    if not pricing_file_path:
        pricing_file_path = get_pricing_file_path(file_path=pricing_file_path)
    with open(pricing_file_path) as fp:
        content = fp.read()
    pricing_data = json.loads(content)
    driver_pricing = pricing_data[driver_type][driver_name]
    if cache_all:
        for driver_type in VALID_PRICING_DRIVER_TYPES:
            pricing = pricing_data.get(driver_type, None)
            if not pricing:
                continue
            PRICING_DATA[driver_type] = pricing
    else:
        set_pricing(driver_type=driver_type, driver_name=driver_name, pricing=driver_pricing)
    return driver_pricing