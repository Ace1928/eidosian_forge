from __future__ import annotations
def strip_transient(nb):
    """Strip transient values that shouldn't be stored in files.

    This should be called in *both* read and write.
    """
    nb.metadata.pop('orig_nbformat', None)
    nb.metadata.pop('orig_nbformat_minor', None)
    nb.metadata.pop('signature', None)
    for cell in nb.cells:
        cell.metadata.pop('trusted', None)
    return nb