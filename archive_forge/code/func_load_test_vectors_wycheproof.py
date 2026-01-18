import os
import re
import json
import errno
import binascii
import warnings
from binascii import unhexlify
from Cryptodome.Util.py3compat import FileNotFoundError
def load_test_vectors_wycheproof(dir_comps, file_name, description, root_tag={}, group_tag={}, unit_tag={}):
    result = []
    try:
        if not test_vectors_available:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_name)
        init_dir = os.path.dirname(pycryptodome_test_vectors.__file__)
        full_file_name = os.path.join(os.path.join(init_dir, *dir_comps), file_name)
        with open(full_file_name) as file_in:
            tv_tree = json.load(file_in)
    except FileNotFoundError:
        warnings.warn('Warning: skipping extended tests for ' + description, UserWarning, stacklevel=2)
        return result

    class TestVector(object):
        pass
    common_root = {}
    for k, v in root_tag.items():
        common_root[k] = v(tv_tree)
    for group in tv_tree['testGroups']:
        common_group = {}
        for k, v in group_tag.items():
            common_group[k] = v(group)
        for test in group['tests']:
            tv = TestVector()
            for k, v in common_root.items():
                setattr(tv, k, v)
            for k, v in common_group.items():
                setattr(tv, k, v)
            tv.id = test['tcId']
            tv.comment = test['comment']
            for attr in ('key', 'iv', 'aad', 'msg', 'ct', 'tag', 'label', 'ikm', 'salt', 'info', 'okm', 'sig', 'public', 'shared'):
                if attr in test:
                    setattr(tv, attr, unhexlify(test[attr]))
            tv.filename = file_name
            for k, v in unit_tag.items():
                setattr(tv, k, v(test))
            tv.valid = test['result'] != 'invalid'
            tv.warning = test['result'] == 'acceptable'
            tv.filename = file_name
            result.append(tv)
    return result