from swiftclient.utils import prt_bytes, split_request_headers
def print_account_stats(items, headers, output_manager):
    exclude_policy_headers = []
    for header_key, header_value in headers.items():
        if header_key.lower().startswith((POLICY_HEADER_PREFIX, PER_POLICY_QUOTA_HEADER_PREFIX)):
            exclude_policy_headers.append(header_key)
    items.extend(headers_to_items(headers, meta_prefix='x-account-meta-', exclude_headers=['content-length', 'date', 'x-account-container-count', 'x-account-object-count', 'x-account-bytes-used', 'x-account-meta-quota-bytes'] + exclude_policy_headers))
    offset = max((len(item) for item, value in items))
    output_manager.print_items(items, offset=offset)