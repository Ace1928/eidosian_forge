from __future__ import unicode_literals
def add_pre_comments(self, comments):
    if not hasattr(self, '_comment'):
        self._comment = [None, None]
    assert self._comment[1] is None
    self._comment[1] = comments