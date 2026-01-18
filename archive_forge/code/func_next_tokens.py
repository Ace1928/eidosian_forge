from abc import ABC, abstractmethod
from typing import List, Optional
def next_tokens(self, current_seq):
    """
        The next possible tokens that will progress the trie, given the current sequence of tokens in `current_seq`.
        """
    start = self.trie
    for current_token in current_seq:
        start = start[current_token]
    next_tokens = list(start.keys())
    return next_tokens