from boto.compat import unquote_str
def bucket_lister(bucket, prefix='', delimiter='', marker='', headers=None, encoding_type=None):
    """
    A generator function for listing keys in a bucket.
    """
    more_results = True
    k = None
    while more_results:
        rs = bucket.get_all_keys(prefix=prefix, marker=marker, delimiter=delimiter, headers=headers, encoding_type=encoding_type)
        for k in rs:
            yield k
        if k:
            marker = rs.next_marker or k.name
        if marker and encoding_type == 'url':
            marker = unquote_str(marker)
        more_results = rs.is_truncated