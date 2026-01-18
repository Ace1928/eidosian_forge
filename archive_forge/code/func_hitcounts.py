from io import BytesIO
from . import osutils, progress, trace
from .i18n import gettext
from .ui import ui_factory
def hitcounts(self, lines):
    """Count the number of hash hits for each tag, for the given lines.

        Hits are weighted according to the number of tags the hash is
        associated with; more tags means that the hash is less rare and should
        tend to be ignored.
        :param lines: The lines to calculate hashes of.
        :return: a dict of {tag: hitcount}
        """
    hits = {}
    for my_hash in self.iter_edge_hashes(lines):
        tags = self.edge_hashes.get(my_hash)
        if tags is None:
            continue
        taglen = len(tags)
        for tag in tags:
            if tag not in hits:
                hits[tag] = 0
            hits[tag] += 1.0 / taglen
    return hits