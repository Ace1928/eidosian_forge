import re
import os.path
from typing import Dict, Union, Optional
from os.path import join as pjoin
def get_size_price(driver_type, driver_name, size_id, region=None):
    """
    Return price for the provided size.

    :type driver_type: ``str``
    :param driver_type: Driver type ('compute' or 'storage')

    :type driver_name: ``str``
    :param driver_name: Driver name

    :type size_id: ``str`` or ``int``
    :param size_id: Unique size ID (can be an integer or a string - depends on
                    the driver)

    :rtype: ``float``
    :return: Size price.
    """
    pricing = get_pricing(driver_type=driver_type, driver_name=driver_name)
    assert pricing is not None
    price = None
    try:
        if region is None:
            price = float(pricing[size_id])
        else:
            price = float(pricing[size_id][region])
    except KeyError:
        price = None
    return price