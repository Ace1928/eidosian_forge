from .. import osutils
def apply_delta_to_source(source, delta_start, delta_end):
    """Extract a delta from source bytes, and apply it."""
    source_size = len(source)
    if delta_start >= source_size:
        raise ValueError('delta starts after source')
    if delta_end > source_size:
        raise ValueError('delta ends after source')
    if delta_start >= delta_end:
        raise ValueError('delta starts after it ends')
    delta_bytes = source[delta_start:delta_end]
    return apply_delta(source, delta_bytes)