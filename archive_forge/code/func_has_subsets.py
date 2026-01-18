from abc import ABC, abstractmethod
from typing import List, Optional
def has_subsets(self, trie, nested_token_ids):
    """
        Returns whether # of leaves == # of words. Otherwise some word is a subset of another.
        """
    leaf_count = self.count_leaves(trie)
    return len(nested_token_ids) != leaf_count