from typing import List, Optional, Union
def in_order(self) -> 'Query':
    """
        Match only documents where the query terms appear in
        the same order in the document.
        i.e. for the query "hello world", we do not match "world hello"
        """
    self._in_order = True
    return self