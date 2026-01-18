from cheroot import makefile
def test_bytes_read():
    """Reader should capture bytes read."""
    sock = MockSocket()
    sock.messages.append(b'foo')
    rfile = makefile.MakeFile(sock, 'r')
    rfile.read()
    assert rfile.bytes_read == 3