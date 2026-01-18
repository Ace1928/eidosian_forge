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
def getDescendentsFromSQL(self, node, type):
    """Get descendents of a node using the database backend.

        This avoids repeated iteration of SQL calls and is therefore much
        quicker than repeatedly calling node.getChildren().
        """
    if nodeCodeOrder.index(type) <= nodeCodeOrder.index(node.type):
        return []
    des_list = []
    if node.type == 'ro':
        for c in node.getChildren():
            for d in self.getDescendentsFromSQL(c, type):
                des_list.append(d)
        return des_list
    cur = self.db_handle.cursor()
    if type != 'px':
        cur.execute('SELECT DISTINCT des.sunid,des.type,des.sccs,description FROM cla,des WHERE cla.' + node.type + '=%s AND cla.' + type + '=des.sunid', node.sunid)
        data = cur.fetchall()
        for d in data:
            if int(d[0]) not in self._sunidDict:
                n = Node(scop=self)
                n.sunid, n.type, n.sccs, n.description = d
                n.sunid = int(n.sunid)
                self._sunidDict[n.sunid] = n
                cur.execute('SELECT parent FROM hie WHERE child=%s', n.sunid)
                n.parent = cur.fetchone()[0]
                cur.execute('SELECT child FROM hie WHERE parent=%s', n.sunid)
                children = []
                for c in cur.fetchall():
                    children.append(c[0])
                n.children = children
            des_list.append(self._sunidDict[int(d[0])])
    else:
        cur.execute('SELECT cla.sunid,sid,pdbid,residues,cla.sccs,type,description,sp FROM cla,des where cla.sunid=des.sunid and cla.' + node.type + '=%s', node.sunid)
        data = cur.fetchall()
        for d in data:
            if int(d[0]) not in self._sunidDict:
                n = Domain(scop=self)
                n.sunid, n.sid, pdbid, n.residues, n.sccs, n.type, n.description, n.parent = d[0:8]
                n.residues = Residues.Residues(n.residues)
                n.residues.pdbid = pdbid
                n.sunid = int(n.sunid)
                self._sunidDict[n.sunid] = n
                self._sidDict[n.sid] = n
            des_list.append(self._sunidDict[int(d[0])])
    return des_list