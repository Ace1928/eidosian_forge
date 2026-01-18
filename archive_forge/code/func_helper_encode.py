from abc import ABC, abstractmethod
from functools import lru_cache
import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple
from typing_extensions import final
from parlai.core.build_data import download, make_dir
from parlai.core.opt import Opt
from parlai.utils.misc import warn_once
from parlai.utils.typing import TShared
import parlai.utils.logging as logging
def helper_encode(self, text: str) -> List[str]:
    """
        Decode list of tokens into text string.

        :param tokens:
            list of tokens
        :param delimiter:
            string delimiter for tokens

        :return text:
            decoded text
        """
    return self.tokenizer.encode(text).tokens