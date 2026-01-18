import json
import uuid
from typing import Optional
import requests
from huggingface_hub import Discussion, HfApi, get_repo_discussions
from .utils import cached_file, logging
def spawn_conversion(token: str, private: bool, model_id: str):
    logger.info('Attempting to convert .bin model on the fly to safetensors.')
    safetensors_convert_space_url = 'https://safetensors-convert.hf.space'
    sse_url = f'{safetensors_convert_space_url}/queue/join'
    sse_data_url = f'{safetensors_convert_space_url}/queue/data'
    hash_data = {'fn_index': 1, 'session_hash': str(uuid.uuid4())}

    def start(_sse_connection, payload):
        for line in _sse_connection.iter_lines():
            line = line.decode()
            if line.startswith('data:'):
                resp = json.loads(line[5:])
                logger.debug(f'Safetensors conversion status: {resp['msg']}')
                if resp['msg'] == 'queue_full':
                    raise ValueError('Queue is full! Please try again.')
                elif resp['msg'] == 'send_data':
                    event_id = resp['event_id']
                    response = requests.post(sse_data_url, stream=True, params=hash_data, json={'event_id': event_id, **payload, **hash_data})
                    response.raise_for_status()
                elif resp['msg'] == 'process_completed':
                    return
    with requests.get(sse_url, stream=True, params=hash_data) as sse_connection:
        data = {'data': [model_id, private, token]}
        try:
            logger.debug('Spawning safetensors automatic conversion.')
            start(sse_connection, data)
        except Exception as e:
            logger.warning(f'Error during conversion: {repr(e)}')