from .. import errors, mail_client, osutils, tests, urlutils
def test_commandline(self):
    claws = mail_client.Claws(None)
    commandline = claws._get_compose_commandline('jrandom@example.org', None, 'file%')
    self.assertEqual(['--compose', 'mailto:jrandom@example.org?', '--attach', 'file%'], commandline)
    commandline = claws._get_compose_commandline('jrandom@example.org', 'Hi there!', None)
    self.assertEqual(['--compose', 'mailto:jrandom@example.org?subject=Hi%20there%21'], commandline)