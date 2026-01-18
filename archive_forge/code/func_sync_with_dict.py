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
def sync_with_dict(self, dict_agent):
    """
        Basically a combination of syncing HF dict with the GPT2 standard.

        It's kinda reversed.

        :param dict_agent:
            Dictionary Agent
        """
    special_tokens = [dict_agent.null_token, dict_agent.start_token, dict_agent.end_token, dict_agent.unk_token]
    dict_agent.tok2ind = {tok: i for tok, i in zip(special_tokens, range(len(special_tokens)))}
    dict_agent.ind2tok = {v: k for k, v in dict_agent.tok2ind.items()}
    for each_token in self.encoder.values():
        dict_agent.add_token(each_token)
        dict_agent.freq[each_token] = 1