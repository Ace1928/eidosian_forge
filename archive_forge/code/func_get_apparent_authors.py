from typing import Dict, List, Optional, Tuple
from . import errors, osutils
def get_apparent_authors(self):
    """Return the apparent authors of this revision.

        If the revision properties contain the names of the authors,
        return them. Otherwise return the committer name.

        The return value will be a list containing at least one element.
        """
    authors = self.properties.get('authors', None)
    if authors is None:
        author = self.properties.get('author', self.committer)
        if author is None:
            return []
        return [author]
    else:
        return authors.split('\n')