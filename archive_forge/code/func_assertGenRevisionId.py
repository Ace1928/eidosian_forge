from ... import tests
from .. import generate_ids
def assertGenRevisionId(self, regex, username, timestamp=None):
    """gen_revision_id should create a revision id matching the regex"""
    revision_id = generate_ids.gen_revision_id(username, timestamp)
    self.assertContainsRe(revision_id, b'^' + regex + b'$')
    self.assertIsInstance(revision_id, bytes)
    revision_id.decode('ascii')