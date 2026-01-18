import unittest
from binascii import a2b_hex, b2a_hex, hexlify
from Cryptodome.Util.py3compat import b
from Cryptodome.Util.strxor import strxor_c
def make_block_tests(module, module_name, test_data, additional_params=dict()):
    tests = []
    extra_tests_added = False
    for i in range(len(test_data)):
        row = test_data[i]
        params = {}
        if len(row) == 3:
            params['plaintext'], params['ciphertext'], params['key'] = row
        elif len(row) == 4:
            params['plaintext'], params['ciphertext'], params['key'], params['description'] = row
        elif len(row) == 5:
            params['plaintext'], params['ciphertext'], params['key'], params['description'], extra_params = row
            params.update(extra_params)
        else:
            raise AssertionError('Unsupported tuple size %d' % (len(row),))
        if not 'mode' in params:
            params['mode'] = 'ECB'
        p2 = params.copy()
        p_key = _extract(p2, 'key')
        p_plaintext = _extract(p2, 'plaintext')
        p_ciphertext = _extract(p2, 'ciphertext')
        p_mode = _extract(p2, 'mode')
        p_description = _extract(p2, 'description', None)
        if p_description is not None:
            description = p_description
        elif p_mode == 'ECB' and (not p2):
            description = 'p=%s, k=%s' % (p_plaintext, p_key)
        else:
            description = 'p=%s, k=%s, %r' % (p_plaintext, p_key, p2)
        name = '%s #%d: %s' % (module_name, i + 1, description)
        params['description'] = name
        params['module_name'] = module_name
        params.update(additional_params)
        if not extra_tests_added:
            tests += [RoundtripTest(module, params), IVLengthTest(module, params), NoDefaultECBTest(module, params), ByteArrayTest(module, params), BlockSizeTest(module, params)]
            extra_tests_added = True
        tests.append(CipherSelfTest(module, params))
    return tests