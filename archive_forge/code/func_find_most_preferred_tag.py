import re
from typing import Dict, Iterable, List
from pip._vendor.packaging.tags import Tag
from pip._internal.exceptions import InvalidWheelFilename
def find_most_preferred_tag(self, tags: List[Tag], tag_to_priority: Dict[Tag, int]) -> int:
    """Return the priority of the most preferred tag that one of the wheel's file
        tag combinations achieves in the given list of supported tags using the given
        tag_to_priority mapping, where lower priorities are more-preferred.

        This is used in place of support_index_min in some cases in order to avoid
        an expensive linear scan of a large list of tags.

        :param tags: the PEP 425 tags to check the wheel against.
        :param tag_to_priority: a mapping from tag to priority of that tag, where
            lower is more preferred.

        :raises ValueError: If none of the wheel's file tags match one of
            the supported tags.
        """
    return min((tag_to_priority[tag] for tag in self.file_tags if tag in tag_to_priority))