from __future__ import annotations
import json
import logging
from typing import Any, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def is_anomalous(self, query: str, **kwargs: Any) -> bool:
    """
        Detect if given text is anomalous from the dataset
        Args:
            query: Text to detect if it is anomaly
        Returns:
            True or False
        """
    vcol = self._vector_index
    vtype = self._vector_type
    embeddings = self._embedding.embed_query(query)
    str_embeddings = [str(f) for f in embeddings]
    qv_comma = ','.join(str_embeddings)
    podstore = self._pod + '.' + self._store
    q = 'select anomalous(' + vcol + ", '" + qv_comma + "', 'type=" + vtype + "')"
    q += ' from ' + podstore
    js = self.run(q)
    if isinstance(js, list) and len(js) == 0:
        return False
    jd = json.loads(js[0])
    if jd['anomalous'] == 'YES':
        return True
    return False