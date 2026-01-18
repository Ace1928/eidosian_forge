import hashlib
from keystone.identity import generator
def generate_public_ID(self, mapping):
    m = hashlib.sha256()
    for key in sorted(mapping.keys()):
        if isinstance(mapping[key], bytes):
            m.update(mapping[key])
        else:
            m.update(mapping[key].encode('utf-8'))
    return m.hexdigest()