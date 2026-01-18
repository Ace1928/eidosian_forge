import logging
from typing import Any, Dict, List, Optional, Union
def query_with_vector(self, vector: List[float]) -> List[Dict[str, Any]]:
    """Perform a vector-based query."""
    vector_query_results = self.dria_client.query(vector, top_n=self.top_n)
    logger.info(f'Vector query results: {vector_query_results}')
    return vector_query_results