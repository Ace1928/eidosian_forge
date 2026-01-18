from rsa._compat import is_integer
def encrypt_int(message, ekey, n):
    """Encrypts a message using encryption key 'ekey', working modulo n"""
    assert_int(message, 'message')
    assert_int(ekey, 'ekey')
    assert_int(n, 'n')
    if message < 0:
        raise ValueError('Only non-negative numbers are supported')
    if message > n:
        raise OverflowError('The message %i is too long for n=%i' % (message, n))
    return pow(message, ekey, n)