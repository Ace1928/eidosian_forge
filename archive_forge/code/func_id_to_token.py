import os
from typing import List, Optional
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
def id_to_token(self, index: int) -> str:
    return self._id_to_token.get(index, self.unk_token)