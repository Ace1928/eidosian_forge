def get_source_stream(source):
    if source == '-' or source is None:
        import sys
        stream = binary_stream(sys.stdin)
    elif source.endswith('.gz'):
        import gzip
        stream = gzip.open(source, 'rb')
    else:
        stream = open(source, 'rb')
    return stream