def test_inet_pton_ipv4_bogus(self):
    with self.assertRaises(socket.error):
        inet_pton(socket.AF_INET, 'blah')