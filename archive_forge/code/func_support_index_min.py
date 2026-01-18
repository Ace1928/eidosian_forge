import re
from typing import Dict, Iterable, List
from pip._vendor.packaging.tags import Tag
from pip._internal.exceptions import InvalidWheelFilename
def support_index_min(self, tags: List[Tag]) -> int:
    """Return the lowest index that one of the wheel's file_tag combinations
        achieves in the given list of supported tags.

        For example, if there are 8 supported tags and one of the file tags
        is first in the list, then return 0.

        :param tags: the PEP 425 tags to check the wheel against, in order
            with most preferred first.

        :raises ValueError: If none of the wheel's file tags match one of
            the supported tags.
        """
    try:
        return next((i for i, t in enumerate(tags) if t in self.file_tags))
    except StopIteration:
        raise ValueError()