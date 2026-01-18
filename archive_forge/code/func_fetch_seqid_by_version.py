import os
from . import BioSeq
from . import Loader
from . import DBUtils
def fetch_seqid_by_version(self, dbid, name):
    """Return the internal id for a sequence using its accession and version.

        Arguments:
         - dbid - the internal id for the sub-database
         - name - the accession of the sequence containing a version number.
           Must correspond to <accession>.<version>

        """
    acc_version = name.split('.')
    if len(acc_version) > 2:
        raise IndexError(f'Bad version {name!r}')
    acc = acc_version[0]
    if len(acc_version) == 2:
        version = acc_version[1]
    else:
        version = '0'
    sql = 'SELECT bioentry_id FROM bioentry WHERE accession = %s AND version = %s'
    fields = [acc, version]
    if dbid:
        sql += ' and biodatabase_id = %s'
        fields.append(dbid)
    self.execute(sql, fields)
    rv = self.cursor.fetchall()
    if not rv:
        raise IndexError(f'Cannot find version {name!r}')
    if len(rv) > 1:
        raise IndexError(f'More than one entry with version {name!r}')
    return rv[0][0]