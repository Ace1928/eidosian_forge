from copy import copy
from ..osutils import contains_linebreaks, contains_whitespace, sha_strings
from ..tree import Tree
def as_sha1(self):
    return sha_strings(self.as_text_lines())