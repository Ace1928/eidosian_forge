from typing import Any, Dict, FrozenSet, Optional, Set, Tuple, Union
import numpy as np
from gym.spaces.space import Space
@property
def character_list(self) -> Tuple[str, ...]:
    """Returns a tuple of characters in the space."""
    return self._char_list