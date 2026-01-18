from datetime import datetime
import logging
import os
from typing import (
import warnings
import numpy as np
from ..core.request import Request, IOMode, InitializationError
from ..core.v3_plugin_api import PluginV3, ImageProperties
@staticmethod
def get_datetime(date: str, time: str) -> Union[datetime, None]:
    """Turn date and time saved by SDT-control into proper datetime object

        Parameters
        ----------
        date
            SPE file date, typically ``metadata["date"]``.
        time
            SPE file date, typically ``metadata["time_local"]``.

        Returns
        -------
        File's datetime if parsing was succsessful, else None.
        """
    try:
        month = __class__.months[date[2:5]]
        return datetime(int(date[5:9]), month, int(date[0:2]), int(time[0:2]), int(time[2:4]), int(time[4:6]))
    except Exception as e:
        logger.info(f'Failed to decode date from SDT-control metadata: {e}.')