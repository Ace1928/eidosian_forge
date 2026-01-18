from typing import Dict, Optional, Tuple
def read_mailmap(f):
    """Read a mailmap.

    Args:
      f: File-like object to read from
    Returns: Iterator over
        ((canonical_name, canonical_email), (from_name, from_email)) tuples
    """
    for line in f:
        line = line.split(b'#')[0]
        line = line.strip()
        if not line:
            continue
        canonical_identity, from_identity = line.split(b'>', 1)
        canonical_identity += b'>'
        if from_identity.strip():
            parsed_from_identity = parse_identity(from_identity)
        else:
            parsed_from_identity = None
        parsed_canonical_identity = parse_identity(canonical_identity)
        yield (parsed_canonical_identity, parsed_from_identity)