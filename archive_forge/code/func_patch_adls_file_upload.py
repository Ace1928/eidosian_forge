import logging
import urllib
from copy import deepcopy
from mlflow.utils import rest_utils
from mlflow.utils.file_utils import read_chunk
def patch_adls_file_upload(sas_url, local_file, start_byte, size, position, headers, is_single):
    """
    Performs an ADLS Azure file create `Patch` operation
    (https://docs.microsoft.com/en-us/rest/api/storageservices/datalakestoragegen2/path/update)

    Args:
        sas_url: A shared access signature URL referring to the Azure ADLS server
            to which the file update command should be issued.
        local_file: The local file to upload
        start_byte: The starting byte of the local file to upload
        size: The number of bytes to upload
        position: Positional offset of the data in the Patch request
        headers: Additional headers to include in the Patch request body
        is_single: Whether this is the only patch operation for this file
    """
    new_params = {'action': 'append', 'position': str(position)}
    if is_single:
        new_params['flush'] = 'true'
    request_url = _append_query_parameters(sas_url, new_params)
    request_headers = {}
    for name, value in headers.items():
        if _is_valid_adls_patch_header(name):
            request_headers[name] = value
        else:
            _logger.debug("Removed unsupported '%s' header for ADLS Gen2 Patch operation", name)
    data = read_chunk(local_file, size, start_byte)
    with rest_utils.cloud_storage_http_request('patch', request_url, data=data, headers=request_headers) as response:
        rest_utils.augmented_raise_for_status(response)