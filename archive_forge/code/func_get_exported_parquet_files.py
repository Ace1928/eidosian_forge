from typing import Any, Dict, List, Optional, Union
from .. import config
from ..exceptions import DatasetsError
from .file_utils import (
from .logging import get_logger
def get_exported_parquet_files(dataset: str, revision: str, token: Optional[Union[str, bool]]) -> List[Dict[str, Any]]:
    """
    Get the dataset exported parquet files
    Docs: https://huggingface.co/docs/datasets-server/parquet
    """
    datasets_server_parquet_url = config.HF_ENDPOINT.replace('://', '://datasets-server.') + '/parquet?dataset='
    try:
        parquet_data_files_response = http_get(url=datasets_server_parquet_url + dataset, temp_file=None, headers=get_authentication_headers_for_url(config.HF_ENDPOINT + f'datasets/{dataset}', token=token), timeout=100.0, max_retries=3)
        parquet_data_files_response.raise_for_status()
        if 'X-Revision' in parquet_data_files_response.headers:
            if parquet_data_files_response.headers['X-Revision'] == revision or revision is None:
                parquet_data_files_response_json = parquet_data_files_response.json()
                if parquet_data_files_response_json.get('partial') is False and (not parquet_data_files_response_json.get('pending', True)) and (not parquet_data_files_response_json.get('failed', True)) and ('parquet_files' in parquet_data_files_response_json):
                    return parquet_data_files_response_json['parquet_files']
                else:
                    logger.debug(f'Parquet export for {dataset} is not completely ready yet.')
            else:
                logger.debug(f"Parquet export for {dataset} is available but outdated (revision='{parquet_data_files_response.headers['X-Revision']}')")
    except Exception as e:
        logger.debug(f'No parquet export for {dataset} available ({type(e).__name__}: {e})')
    raise DatasetsServerError('No exported Parquet files available.')