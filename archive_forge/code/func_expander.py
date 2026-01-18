from typing import List, Optional, Union
def expander(self, expander: str) -> 'Query':
    """
        Add a expander field to the query.

        - **expander** - the name of the expander
        """
    self._expander = expander
    return self