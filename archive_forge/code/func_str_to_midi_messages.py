import json
import random
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor, log
from typing import Dict, Iterator, List, Optional, Tuple
import mido
def str_to_midi_messages(utils: VocabUtils, data: str) -> Iterator[mido.Message]:
    state = None
    for token in data.split(' '):
        for msg, new_state in token_to_midi_message(utils, token, state):
            state = new_state
            if msg is not None:
                yield msg