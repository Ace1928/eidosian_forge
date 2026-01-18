def test_inet_pton_ipv6_bogus(self):
    with self.assertRaises(socket.error):
        inet_pton(socket.AF_INET6, 'blah')