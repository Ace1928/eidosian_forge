from typing import Any, Dict, FrozenSet, Optional, Set, Tuple, Union
import numpy as np
from gym.spaces.space import Space
def character_index(self, char: str) -> np.int32:
    """Returns a unique index for each character in the space's character set."""
    return self._char_index[char]