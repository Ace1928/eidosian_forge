from io import BytesIO
from .. import osutils
def parse_roundtripping_metadata(text):
    """Parse Bazaar roundtripping metadata."""
    ret = CommitSupplement()
    f = BytesIO(text)
    for l in f.readlines():
        key, value = l.split(b':', 1)
        if key == b'revision-id':
            ret.revision_id = value.strip()
        elif key == b'parent-ids':
            ret.explicit_parent_ids = tuple(value.strip().split(b' '))
        elif key == b'testament3-sha1':
            ret.verifiers[b'testament3-sha1'] = value.strip()
        elif key.startswith(b'property-'):
            name = key[len(b'property-'):]
            if name not in ret.properties:
                ret.properties[name] = value[1:].rstrip(b'\n')
            else:
                ret.properties[name] += b'\n' + value[1:].rstrip(b'\n')
        else:
            raise ValueError
    return ret