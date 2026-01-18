import json
import logging
from time import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from langchain_core._api import deprecated
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
@staticmethod
def connect_to_elasticsearch(*, es_url: Optional[str]=None, cloud_id: Optional[str]=None, api_key: Optional[str]=None, username: Optional[str]=None, password: Optional[str]=None) -> 'Elasticsearch':
    try:
        import elasticsearch
    except ImportError:
        raise ImportError('Could not import elasticsearch python package. Please install it with `pip install elasticsearch`.')
    if es_url and cloud_id:
        raise ValueError('Both es_url and cloud_id are defined. Please provide only one.')
    connection_params: Dict[str, Any] = {}
    if es_url:
        connection_params['hosts'] = [es_url]
    elif cloud_id:
        connection_params['cloud_id'] = cloud_id
    else:
        raise ValueError('Please provide either elasticsearch_url or cloud_id.')
    if api_key:
        connection_params['api_key'] = api_key
    elif username and password:
        connection_params['basic_auth'] = (username, password)
    es_client = elasticsearch.Elasticsearch(**connection_params, headers={'user-agent': ElasticsearchChatMessageHistory.get_user_agent()})
    try:
        es_client.info()
    except Exception as err:
        logger.error(f'Error connecting to Elasticsearch: {err}')
        raise err
    return es_client