import warnings
from Bio import BiopythonWarning
from Bio import MissingPythonDependencyError
from Bio.motifs import jaspar, matrix
def fetch_motifs(self, collection=JASPAR_DFLT_COLLECTION, tf_name=None, tf_class=None, tf_family=None, matrix_id=None, tax_group=None, species=None, pazar_id=None, data_type=None, medline=None, min_ic=0, min_length=0, min_sites=0, all=False, all_versions=False):
    """Fetch jaspar.Record (list) of motifs using selection criteria.

        Arguments::

            Except where obvious, all selection criteria arguments may be
            specified as a single value or a list of values. Motifs must
            meet ALL the specified selection criteria to be returned with
            the precedent exceptions noted below.

            all         - Takes precedent of all other selection criteria.
                          Every motif is returned. If 'all_versions' is also
                          specified, all versions of every motif are returned,
                          otherwise just the latest version of every motif is
                          returned.
            matrix_id   - Takes precedence over all other selection criteria
                          except 'all'.  Only motifs with the given JASPAR
                          matrix ID(s) are returned. A matrix ID may be
                          specified as just a base ID or full JASPAR IDs
                          including version number. If only a base ID is
                          provided for specific motif(s), then just the latest
                          version of those motif(s) are returned unless
                          'all_versions' is also specified.
            collection  - Only motifs from the specified JASPAR collection(s)
                          are returned. NOTE - if not specified, the collection
                          defaults to CORE for all other selection criteria
                          except 'all' and 'matrix_id'. To apply the other
                          selection criteria across all JASPAR collections,
                          explicitly set collection=None.
            tf_name     - Only motifs with the given name(s) are returned.
            tf_class    - Only motifs of the given TF class(es) are returned.
            tf_family   - Only motifs from the given TF families are returned.
            tax_group   - Only motifs belonging to the given taxonomic
                          supergroups are returned (e.g. 'vertebrates',
                          'insects', 'nematodes' etc.)
            species     - Only motifs derived from the given species are
                          returned.  Species are specified as taxonomy IDs.
            data_type   - Only motifs generated with the given data type (e.g.
                          ('ChIP-seq', 'PBM', 'SELEX' etc.) are returned.
                          NOTE - must match exactly as stored in the database.
            pazar_id    - Only motifs with the given PAZAR TF ID are returned.
            medline     - Only motifs with the given medline (PubmMed IDs) are
                          returned.
            min_ic      - Only motifs whose profile matrices have at least this
                          information content (specificty) are returned.
            min_length  - Only motifs whose profiles are of at least this
                          length are returned.
            min_sites   - Only motifs compiled from at least these many binding
                          sites are returned.
            all_versions- Unless specified, just the latest version of motifs
                          determined by the other selection criteria are
                          returned. Otherwise all versions of the selected
                          motifs are returned.

        Returns:
            - A Bio.motifs.jaspar.Record (list) of motifs.

        """
    int_ids = self._fetch_internal_id_list(collection=collection, tf_name=tf_name, tf_class=tf_class, tf_family=tf_family, matrix_id=matrix_id, tax_group=tax_group, species=species, pazar_id=pazar_id, data_type=data_type, medline=medline, all=all, all_versions=all_versions)
    record = jaspar.Record()
    '\n        Now further filter motifs returned above based on any specified\n        matrix specific criteria.\n        '
    for int_id in int_ids:
        motif = self._fetch_motif_by_internal_id(int_id)
        if min_ic:
            if motif.pssm.mean() < min_ic:
                continue
        if min_length:
            if motif.length < min_length:
                continue
        '\n            Filter motifs to those composed of at least this many sites.\n            The perl TFBS module assumes column sums may be different but\n            this should be strictly enforced here we will ignore this and\n            just use the first column sum.\n            '
        if min_sites:
            num_sites = sum((motif.counts[nt][0] for nt in motif.alphabet))
            if num_sites < min_sites:
                continue
        record.append(motif)
    return record