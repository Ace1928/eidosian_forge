import os
from . import BioSeq
from . import Loader
from . import DBUtils
def fetch_seqid_by_accession(self, dbid, name):
    """Return the internal id for a sequence using its accession.

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
    self.execute(sql, fields)
    rv = self.cursor.fetchall()
    if not rv:
        raise IndexError(f'Cannot find accession {name!r}')
    if len(rv) > 1:
        raise IndexError(f'More than one entry with accession {name!r}')
    return rv[0][0]