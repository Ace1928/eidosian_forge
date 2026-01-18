import os
import re
import sqlite3
import subprocess
import sys
from optparse import OptionParser
from rdkit import Chem
def index_hydrogen_change():
    cursor.execute('SELECT context_smi FROM context_table')
    res = cursor.fetchall()
    for row in res:
        key = row[0]
        attachments = key.count('*')
        if attachments == 1:
            smi = str(key)
            smi = re.sub('\\[\\*\\:1\\]', '[H]', smi)
            temp = Chem.MolFromSmiles(smi)
            if temp is None:
                sys.stderr.write('Error with key: %s, Added H: %s\n' % (key, smi))
            else:
                c_smi = Chem.MolToSmiles(temp, isomericSmiles=True)
                cursor.execute('SELECT cmpd_id FROM cmpd_smisp WHERE smiles = ?', (c_smi,))
                cmpd_id = cursor.fetchone()
                if cmpd_id:
                    core = '[*:1][H]'
                    cmpd_id = cmpd_id[0]
                    key_size = temp.GetNumAtoms() - 1
                    add_to_db(cmpd_id, core, key, key_size)