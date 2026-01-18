import re
def make_format(self, cut=(), title='', nc=(), s1=''):
    """Virtual method used for formatting results.

        Virtual method.
        Here to be pointed to one of the _make_* methods.
        You can as well create a new method and point make_format to it.
        """
    return self._make_list(cut, title, nc, s1)