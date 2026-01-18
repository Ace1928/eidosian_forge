from typing import Any, Dict, FrozenSet, Optional, Set, Tuple, Union
import numpy as np
from gym.spaces.space import Space
@property
def character_set(self) -> FrozenSet[str]:
    """Returns the character set for the space."""
    return self._char_set