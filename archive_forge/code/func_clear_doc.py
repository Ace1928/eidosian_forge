from typing import TYPE_CHECKING, Dict, List, Optional, Set
from docutils import nodes
from sphinx.environment import BuildEnvironment
def clear_doc(self, app: 'Sphinx', env: BuildEnvironment, docname: str) -> None:
    """Remove specified data of a document.

        This method is called on the removal of the document."""
    raise NotImplementedError