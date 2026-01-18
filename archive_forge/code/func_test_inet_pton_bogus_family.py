def test_inet_pton_bogus_family(self):
    i = int(socket.AF_INET6)
    while True:
        if i != socket.AF_INET and i != socket.AF_INET6:
            break
        i += 100
    with self.assertRaises(socket.error):
        inet_pton(i, '127.0.0.1')