import os
import re
import sqlite3
import subprocess
import sys
from optparse import OptionParser
from rdkit import Chem
def add_to_db(id, core, context, context_size):
    cursor.execute('INSERT OR IGNORE INTO context_table(context_smi) VALUES(?)', (context,))
    the_id_of_the_row = None
    cursor.execute('SELECT context_id FROM context_table WHERE context_smi = ?', (context,))
    the_id_of_the_row = cursor.fetchone()[0]
    core_ni = re.sub('\\[\\*\\:1\\]', '[*]', core)
    core_ni = re.sub('\\[\\*\\:2\\]', '[*]', core_ni)
    core_ni = re.sub('\\[\\*\\:3\\]', '[*]', core_ni)
    cursor.execute('INSERT INTO core_table(context_id, cmpd_id, core_smi, core_smi_ni) VALUES(?,?,?,?);', (the_id_of_the_row, id, core, core_ni))
    cursor.execute('select context_size from context_table where context_id = ?', (the_id_of_the_row,))
    heavy_calculated = cursor.fetchone()
    if heavy_calculated[0] is None:
        cursor.execute('update context_table set context_size = ? where context_id = ? ', (context_size, the_id_of_the_row))