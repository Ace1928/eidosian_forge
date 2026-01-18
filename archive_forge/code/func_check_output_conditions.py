from __future__ import annotations
from traitlets import Set, Unicode
from .base import Preprocessor
def check_output_conditions(self, output, resources, cell_index, output_index):
    """
        Checks that an output has a tag that indicates removal.

        Returns: Boolean.
        True means output should *not* be removed.
        """
    return not self.remove_single_output_tags.intersection(output.get('metadata', {}).get('tags', []))