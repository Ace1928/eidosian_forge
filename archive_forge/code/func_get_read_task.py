import json
import logging
import time
from typing import List, Optional
from urllib.parse import urlencode, urljoin
import pyarrow
import requests
from ray.data.block import BlockMetadata
from ray.data.datasource.datasource import Datasource, ReadTask
from ray.util.annotations import PublicAPI
def get_read_task(task_index, parallelism):
    chunk_index_list = list(range(task_index, parallelism, num_chunks))
    num_rows = sum((chunks[chunk_index]['row_count'] for chunk_index in chunk_index_list))
    size_bytes = sum((chunks[chunk_index]['byte_count'] for chunk_index in chunk_index_list))
    metadata = BlockMetadata(num_rows=num_rows, size_bytes=size_bytes, schema=None, input_files=None, exec_stats=None)

    def read_fn():
        for chunk_index in chunk_index_list:
            chunk_info = chunks[chunk_index]
            row_offset_param = urlencode({'row_offset': chunk_info['row_offset']})
            resolve_external_link_url = urljoin(url_base, f'{statement_id}/result/chunks/{chunk_index}?{row_offset_param}')
            resolve_response = requests.get(resolve_external_link_url, auth=req_auth, headers=req_headers)
            resolve_response.raise_for_status()
            external_url = resolve_response.json()['external_links'][0]['external_link']
            raw_response = requests.get(external_url, auth=None, headers=None)
            raw_response.raise_for_status()
            arrow_table = pyarrow.ipc.open_stream(raw_response.content).read_all()
            yield arrow_table
    return ReadTask(read_fn=read_fn, metadata=metadata)