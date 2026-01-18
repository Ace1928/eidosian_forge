from gitdb.util import (
from gitdb.utils.encoding import force_text
from gitdb.exc import (
from itertools import chain
from functools import reduce
def db_path(self, rela_path):
    """
        :return: the given relative path relative to our database root, allowing
            to pontentially access datafiles"""
    return join(self._root_path, force_text(rela_path))