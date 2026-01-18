from __future__ import unicode_literals
def split_comment(self):
    """ split the post part of a comment, and return it
        as comment to be added. Delete second part if [None, None]
         abc:  # this goes to sequence
           # this goes to first element
           - first element
        """
    comment = self.comment
    if comment is None or comment[0] is None:
        return None
    ret_val = [comment[0], None]
    if comment[1] is None:
        delattr(self, '_comment')
    return ret_val