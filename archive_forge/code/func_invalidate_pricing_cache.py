import re
import os.path
from typing import Dict, Union, Optional
from os.path import join as pjoin
def invalidate_pricing_cache():
    """
    Invalidate pricing cache for all the drivers.
    """
    PRICING_DATA['compute'] = {}
    PRICING_DATA['storage'] = {}