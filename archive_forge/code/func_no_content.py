from typing import List, Optional, Union
def no_content(self) -> 'Query':
    """Set the query to only return ids and not the document content."""
    self._no_content = True
    return self