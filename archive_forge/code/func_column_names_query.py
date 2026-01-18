from typing import Any, Dict, Optional, Sequence
def column_names_query(self, query: str) -> str:
    """
        Get a query that gives the names of columns that `query` would produce.

        Parameters
        ----------
        query : str
            The SQL query to check.

        Returns
        -------
        str
        """
    return f'SELECT * FROM ({query}) AS _MODIN_COUNT_QUERY WHERE 1 = 0'