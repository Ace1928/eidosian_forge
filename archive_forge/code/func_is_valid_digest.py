from __future__ import (absolute_import, division, print_function)
import re
def is_valid_digest(digest):
    digest_algorithm_size = dict(sha256=64, sha384=96, sha512=128)
    m = re.match('[a-zA-Z0-9-_+.]+:[a-fA-F0-9]+', digest)
    if not m:
        return 'Docker digest does not match expected format %s' % digest
    idx = digest.find(':')
    if idx < 0 or idx == len(digest) - 1:
        return 'Invalid docker digest %s, no hex value define' % digest
    algorithm = digest[:idx]
    if algorithm not in digest_algorithm_size:
        return 'Unsupported digest algorithm value %s for digest %s' % (algorithm, digest)
    hex_value = digest[idx + 1:]
    if len(hex_value) != digest_algorithm_size.get(algorithm):
        return 'Invalid length for digest hex expected %d found %d (digest is %s)' % (digest_algorithm_size.get(algorithm), len(hex_value), digest)