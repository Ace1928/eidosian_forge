import struct
from passlib.utils import repeat_string
def eks_salted_expand(self, key_words, salt_words):
    """perform EKS' salted version of Blowfish keyschedule setup"""
    assert len(key_words) >= 18, 'key_words must be at least as large as P'
    salt_size = len(salt_words)
    assert salt_size, 'salt_words must not be empty'
    assert not salt_size & 1, 'salt_words must have even length'
    P, S, encipher = (self.P, self.S, self.encipher)
    i = 0
    while i < 18:
        P[i] ^= key_words[i]
        i += 1
    s = i = l = r = 0
    while i < 18:
        l ^= salt_words[s]
        r ^= salt_words[s + 1]
        s += 2
        if s == salt_size:
            s = 0
        P[i], P[i + 1] = l, r = encipher(l, r)
        i += 2
    for box in S:
        i = 0
        while i < 256:
            l ^= salt_words[s]
            r ^= salt_words[s + 1]
            s += 2
            if s == salt_size:
                s = 0
            box[i], box[i + 1] = l, r = encipher(l, r)
            i += 2