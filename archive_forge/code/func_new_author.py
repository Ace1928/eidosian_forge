from __future__ import annotations
from nbformat._struct import Struct
def new_author(name=None, email=None, affiliation=None, url=None):
    """Create a new author."""
    author = NotebookNode()
    if name is not None:
        author.name = str(name)
    if email is not None:
        author.email = str(email)
    if affiliation is not None:
        author.affiliation = str(affiliation)
    if url is not None:
        author.url = str(url)
    return author