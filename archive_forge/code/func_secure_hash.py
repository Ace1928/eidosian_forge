from __future__ import (absolute_import, division, print_function)
import os
from hashlib import sha1
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_bytes
def secure_hash(filename, hash_func=sha1):
    """ Return a secure hash hex digest of local file, None if file is not present or a directory. """
    if not os.path.exists(to_bytes(filename, errors='surrogate_or_strict')) or os.path.isdir(to_bytes(filename, errors='strict')):
        return None
    digest = hash_func()
    blocksize = 64 * 1024
    try:
        infile = open(to_bytes(filename, errors='surrogate_or_strict'), 'rb')
        block = infile.read(blocksize)
        while block:
            digest.update(block)
            block = infile.read(blocksize)
        infile.close()
    except IOError as e:
        raise AnsibleError('error while accessing the file %s, error was: %s' % (filename, e))
    return digest.hexdigest()