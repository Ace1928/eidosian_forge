from typing import List, Optional, Union
def limit_ids(self, *ids) -> 'Query':
    """Limit the results to a specific set of pre-known document
        ids of any length."""
    self._ids = ids
    return self