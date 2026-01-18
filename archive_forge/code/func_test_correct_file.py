import os
import pytest
from nltk.twitter import Authenticate
def test_correct_file(self, auth):
    """Test that a proper file succeeds and is read correctly"""
    oauth = auth.load_creds(subdir=self.subdir)
    assert auth.creds_fullpath == os.path.join(self.subdir, auth.creds_file)
    assert auth.creds_file == 'credentials.txt'
    assert oauth['app_key'] == 'a'