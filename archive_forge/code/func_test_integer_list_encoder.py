import re
import string
def test_integer_list_encoder(tries=1000, length=100, max_entry=2 ** 90):
    import random
    tests = 0
    for i in range(tries):
        entries = [random.randrange(-max_entry, max_entry) for i in range(length)]
        entries += [random.randrange(-15, 16) for i in range(length)]
        random.shuffle(entries)
        assert decode_integer_list(encode_integer_list(entries)) == entries
        tests += 1
    print('Tested encode/decode on %d lists of integers' % tests)