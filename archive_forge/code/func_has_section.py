import os
import sys
from contextlib import suppress
from typing import (
from .file import GitFile
def has_section(self, name: Section) -> bool:
    """Check if a specified section exists.

        Args:
          name: Name of section to check for
        Returns:
          boolean indicating whether the section exists
        """
    return name in self.sections()