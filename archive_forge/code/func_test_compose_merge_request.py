from .. import errors, mail_client, osutils, tests, urlutils
def test_compose_merge_request(self):
    client = DefaultMailDummyClient()
    to = 'a@b.com'
    subject = '[MERGE]'
    directive = ('directive',)
    basename = 'merge'
    client.compose_merge_request(to, subject, directive, basename=basename)
    dummy_client = client.client
    self.assertEqual(dummy_client.args, (to, subject, directive))
    self.assertEqual(dummy_client.kwargs, {'basename': basename, 'body': None})