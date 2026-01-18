import os
from . import BioSeq
from . import Loader
from . import DBUtils
def get_Seq_by_ver(self, name):
    """Get a DBSeqRecord object by version number.

        Example: seq_rec = db.get_Seq_by_ver('X77802.1')

        The name of this method is misleading since it returns a DBSeqRecord
        rather than a Seq object, and presumably was to mirror BioPerl.
        """
    seqid = self.adaptor.fetch_seqid_by_version(self.dbid, name)
    return BioSeq.DBSeqRecord(self.adaptor, seqid)