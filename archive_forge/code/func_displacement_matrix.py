import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union
from .. import mult, orient
from ._font import Font
def displacement_matrix(self, word: Union[str, None]=None, TD_offset: float=0.0) -> List[float]:
    """
        Text displacement matrix

        Args:
            word (str, optional): Defaults to None in which case self.txt displacement is
                returned.
            TD_offset (float, optional): translation applied by TD operator. Defaults to 0.0.
        """
    word = word if word is not None else self.txt
    return [1.0, 0.0, 0.0, 1.0, self.word_tx(word, TD_offset), 0.0]