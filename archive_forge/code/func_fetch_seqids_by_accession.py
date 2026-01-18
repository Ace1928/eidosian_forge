import os
from . import BioSeq
from . import Loader
from . import DBUtils
def fetch_seqids_by_accession(self, dbid, name):
    """Return a list internal ids using an accession.

        Arguments:
         - dbid - the internal id for the sub-database
         - name - the accession of the sequence. Corresponds to the
           accession column of the bioentry table of the SQL schema

        """
    sql = 'select bioentry_id from bioentry where accession = %s'
    fields = [name]
    if dbid:
        sql += ' and biodatabase_id = %s'
        fields.append(dbid)
    return self.execute_and_fetch_col0(sql, fields)