import abc
import os
import re
from googlecloudsdk.core.util import files
def get_cpu_count(self) -> int:
    """Returns the number of logical CPUs in the system.

    Logical CPU is the number of threads that the OS can schedule work on.
    Includes physical cores and hyper-threaded cores.
    """
    return os.cpu_count()