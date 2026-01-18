import re
import os.path
from typing import Dict, Union, Optional
from os.path import join as pjoin
def set_pricing(driver_type, driver_name, pricing):
    """
    Populate the driver pricing dictionary.

    :type driver_type: ``str``
    :param driver_type: Driver type ('compute' or 'storage')

    :type driver_name: ``str``
    :param driver_name: Driver name

    :type pricing: ``dict``
    :param pricing: Dictionary where a key is a size ID and a value is a price.
    """
    PRICING_DATA[driver_type][driver_name] = pricing