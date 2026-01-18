from os import getpid
from typing import Dict, List, Mapping, Optional, Sequence
from attrs import Factory, define
def inheritedDescriptors(self) -> List[int]:
    """
        @return: The configured descriptors.
        """
    return list(self._descriptors)