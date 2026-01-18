from ... import tests
from .. import generate_ids
def assertGenFileId(self, regex, filename):
    """gen_file_id should create a file id matching the regex.

        The file id should be ascii, and should be an 8-bit string
        """
    file_id = generate_ids.gen_file_id(filename)
    self.assertContainsRe(file_id, b'^' + regex + b'$')
    self.assertIsInstance(file_id, bytes)
    file_id.decode('ascii')