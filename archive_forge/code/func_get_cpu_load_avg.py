import abc
import os
import re
from googlecloudsdk.core.util import files
def get_cpu_load_avg(self) -> float:
    """Returns the average CPU load during last 1-minute."""
    return os.getloadavg()[0]