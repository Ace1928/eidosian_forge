import json
import logging
import numbers
from hashlib import sha1
from typing import Any, Dict, Iterable, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def inner_embedding_query(self, embedding: List[float], search_filter: Optional[Dict[str, Any]]=None, k: int=4) -> Dict[str, Any]:

    def generate_filter_query() -> str:
        if search_filter is None:
            return ''
        filter_clause = ' AND '.join([create_filter(md_key, md_value) for md_key, md_value in search_filter.items()])
        return filter_clause

    def create_filter(md_key: str, md_value: Any) -> str:
        md_filter_expr = self.config.field_name_mapping[md_key]
        if md_filter_expr is None:
            return ''
        expr = md_filter_expr.split(',')
        if len(expr) != 2:
            logger.error(f'filter {md_filter_expr} express is not correct, must contain mapping field and operator.')
            return ''
        md_filter_key = expr[0].strip()
        md_filter_operator = expr[1].strip()
        if isinstance(md_value, numbers.Number):
            return f'{md_filter_key} {md_filter_operator} {md_value}'
        return f'{md_filter_key}{md_filter_operator}"{md_value}"'

    def search_data() -> Dict[str, Any]:
        request = QueryRequest(table_name=self.config.table_name, namespace=self.config.namespace, vector=embedding, include_vector=True, output_fields=self.config.output_fields, filter=generate_filter_query(), top_k=k)
        query_result = self.ha3_engine_client.query(request)
        return json.loads(query_result.body)
    from alibabacloud_ha3engine_vector.models import QueryRequest
    try:
        json_response = search_data()
        if 'errorCode' in json_response and 'errorMsg' in json_response and (len(json_response['errorMsg']) > 0):
            logger.error(f'query {self.config.endpoint} {self.config.instance_id} failed:{json_response['errorMsg']}.')
        else:
            return json_response
    except Exception as e:
        logger.error(f'query instance endpoint:{self.config.endpoint} instance_id:{self.config.instance_id} failed.', e)
    return {}