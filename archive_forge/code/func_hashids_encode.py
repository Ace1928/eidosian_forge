from __future__ import absolute_import, division, print_function
from ansible.errors import (
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.collections import is_sequence
def hashids_encode(nums, salt=None, alphabet=None, min_length=None):
    """Generates a YouTube-like hash from a sequence of ints

       :nums: Sequence of one or more ints to hash
       :salt: String to use as salt when hashing
       :alphabet: String of 16 or more unique characters to produce a hash
       :min_length: Minimum length of hash produced
    """
    hashids = initialize_hashids(salt=salt, alphabet=alphabet, min_length=min_length)
    if not is_sequence(nums):
        nums = [nums]
    try:
        hashid = hashids.encode(*nums)
    except TypeError as e:
        raise AnsibleFilterTypeError('Data to encode must by a tuple or list of ints: %s' % to_native(e))
    return hashid