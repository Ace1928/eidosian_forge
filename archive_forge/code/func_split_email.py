import stat
from typing import Dict, Tuple
from fastimport import commands, parser, processor
from fastimport import errors as fastimport_errors
from .index import commit_tree
from .object_store import iter_tree_contents
from .objects import ZERO_SHA, Blob, Commit, Tag
def split_email(text):
    name, email = text.rsplit(b' <', 1)
    return (name, email.rstrip(b'>'))