from typing import List, Union
def dialect(self, dialect: int) -> 'AggregateRequest':
    """
        Add a dialect field to the aggregate command.

        - **dialect** - dialect version to execute the query under
        """
    self._dialect = dialect
    return self