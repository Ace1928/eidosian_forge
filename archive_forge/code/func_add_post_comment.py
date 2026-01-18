from __future__ import unicode_literals
def add_post_comment(self, comment):
    if not hasattr(self, '_comment'):
        self._comment = [None, None]
    self._comment[0] = comment