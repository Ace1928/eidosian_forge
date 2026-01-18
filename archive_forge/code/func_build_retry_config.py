import copy
from botocore.utils import merge_dicts
def build_retry_config(endpoint_prefix, retry_model, definitions, client_retry_config=None):
    service_config = retry_model.get(endpoint_prefix, {})
    resolve_references(service_config, definitions)
    final_retry_config = {'__default__': copy.deepcopy(retry_model.get('__default__', {}))}
    resolve_references(final_retry_config, definitions)
    merge_dicts(final_retry_config, service_config)
    if client_retry_config is not None:
        _merge_client_retry_config(final_retry_config, client_retry_config)
    return final_retry_config