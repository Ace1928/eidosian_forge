import warnings
from Bio import BiopythonWarning
from Bio import MissingPythonDependencyError
from Bio.motifs import jaspar, matrix
def fetch_motif_by_id(self, id):
    """Fetch a single JASPAR motif from the DB by its JASPAR matrix ID.

        Example id 'MA0001.1'.

        Arguments:
         - id - JASPAR matrix ID. This may be a fully specified ID including
                the version number (e.g. MA0049.2) or just the base ID (e.g.
                MA0049). If only a base ID is provided, the latest version is
                returned.

        Returns:
         - A Bio.motifs.jaspar.Motif object

        **NOTE:** The perl TFBS module allows you to specify the type of matrix
        to return (PFM, PWM, ICM) but matrices are always stored in JASPAR as
        PFMs so this does not really belong here. Once a PFM is fetched the
        pwm() and pssm() methods can be called to return the normalized and
        log-odds matrices.

        """
    base_id, version = jaspar.split_jaspar_id(id)
    if not version:
        version = self._fetch_latest_version(base_id)
    int_id = None
    if version:
        int_id = self._fetch_internal_id(base_id, version)
    motif = None
    if int_id:
        motif = self._fetch_motif_by_internal_id(int_id)
    return motif