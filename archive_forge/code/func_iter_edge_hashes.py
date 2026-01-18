from io import BytesIO
from . import osutils, progress, trace
from .i18n import gettext
from .ui import ui_factory
@staticmethod
def iter_edge_hashes(lines):
    """Iterate through the hashes of line pairs (which make up an edge).

        The hash is truncated using a modulus to avoid excessive memory
        consumption by the hitscount dict.  A modulus of 10Mi means that the
        maximum number of keys is 10Mi.  (Keys are normally 32 bits, e.g.
        4 Gi)
        """
    modulus = 1024 * 1024 * 10
    for n in range(len(lines)):
        yield (hash(tuple(lines[n:n + 2])) % modulus)