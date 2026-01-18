import re
import os.path
from typing import Dict, Union, Optional
from os.path import join as pjoin
def clear_pricing_data():
    """
    Invalidate pricing cache for all the drivers.

    Note: This method does the same thing as invalidate_pricing_cache and is
    here for backward compatibility reasons.
    """
    invalidate_pricing_cache()