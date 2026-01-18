import os
from . import BioSeq
from . import Loader
from . import DBUtils
def get_Seqs_by_acc(self, name):
    """Get a list of DBSeqRecord objects by accession number.

        Example: seq_recs = db.get_Seq_by_acc('X77802')

        The name of this method is misleading since it returns a list of
        DBSeqRecord objects rather than a list of Seq objects, and presumably
        was to mirror BioPerl.
        """
    seqids = self.adaptor.fetch_seqids_by_accession(self.dbid, name)
    return [BioSeq.DBSeqRecord(self.adaptor, seqid) for seqid in seqids]