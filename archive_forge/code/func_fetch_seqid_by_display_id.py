import os
from . import BioSeq
from . import Loader
from . import DBUtils
def fetch_seqid_by_display_id(self, dbid, name):
    """Return the internal id for a sequence using its display id.

        Arguments:
         - dbid - the internal id for the sub-database
         - name - the name of the sequence. Corresponds to the
           name column of the bioentry table of the SQL schema

        """
    sql = 'select bioentry_id from bioentry where name = %s'
    fields = [name]
    if dbid:
        sql += ' and biodatabase_id = %s'
        fields.append(dbid)
    self.execute(sql, fields)
    rv = self.cursor.fetchall()
    if not rv:
        raise IndexError(f'Cannot find display id {name!r}')
    if len(rv) > 1:
        raise IndexError(f'More than one entry with display id {name!r}')
    return rv[0][0]