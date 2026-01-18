import hashlib
import zlib
from .. import estimate_compressed_size, tests
def get_slightly_random_content(self, length, seed=b''):
    """We generate some hex-data that can be seeded.

        The output should be deterministic, but the data stream is effectively
        random.
        """
    h = hashlib.md5(seed)
    hex_content = []
    count = 0
    while count < length:
        b = h.hexdigest().encode('ascii')
        hex_content.append(b)
        h.update(b)
        count += len(b)
    return b''.join(hex_content)[:length]