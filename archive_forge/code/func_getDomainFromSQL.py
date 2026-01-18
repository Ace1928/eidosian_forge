import os
import re
from urllib.parse import urlencode
from urllib.request import urlopen
from . import Des
from . import Cla
from . import Hie
from . import Residues
from Bio import SeqIO
from Bio.Seq import Seq
def getDomainFromSQL(self, sunid=None, sid=None):
    """Load a node from the SQL backend using sunid or sid."""
    if sunid is None and sid is None:
        return None
    cur = self.db_handle.cursor()
    if sid:
        cur.execute('SELECT sunid FROM cla WHERE sid=%s', sid)
        res = cur.fetchone()
        if res is None:
            return None
        sunid = res[0]
    cur.execute('SELECT * FROM des WHERE sunid=%s', sunid)
    data = cur.fetchone()
    if data is not None:
        n = None
        if data[1] != 'px':
            n = Node(scop=self)
            cur.execute('SELECT child FROM hie WHERE parent=%s', sunid)
            children = []
            for c in cur.fetchall():
                children.append(c[0])
            n.children = children
        else:
            n = Domain(scop=self)
            cur.execute('select sid, residues, pdbid from cla where sunid=%s', sunid)
            n.sid, n.residues, pdbid = cur.fetchone()
            n.residues = Residues.Residues(n.residues)
            n.residues.pdbid = pdbid
            self._sidDict[n.sid] = n
        n.sunid, n.type, n.sccs, n.description = data
        if data[1] != 'ro':
            cur.execute('SELECT parent FROM hie WHERE child=%s', sunid)
            n.parent = cur.fetchone()[0]
        n.sunid = int(n.sunid)
        self._sunidDict[n.sunid] = n