import re
from typing import Dict, Iterable, List
from pip._vendor.packaging.tags import Tag
from pip._internal.exceptions import InvalidWheelFilename
def get_formatted_file_tags(self) -> List[str]:
    """Return the wheel's tags as a sorted list of strings."""
    return sorted((str(tag) for tag in self.file_tags))