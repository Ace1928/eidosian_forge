import os
import re
import sqlite3
import subprocess
import sys
from optparse import OptionParser
from indexing import cansmirk, heavy_atom_count
from rfrag import fragment_mol
def print_smallest_change_mmp(db_results, cmpd_id, query_size):
    uniq_list = {}
    for r in db_results:
        if r[0] != cmpd_id:
            if r[0] not in uniq_list:
                uniq_list[r[0]] = r
            elif r[3] > uniq_list[r[0]][3]:
                uniq_list[r[0]] = r
    for key, value in uniq_list.items():
        size_of_change = query_size - value[3]
        if use_ratio:
            if float(size_of_change) / query_size <= ratio:
                cursor.execute('SELECT smiles FROM cmpd_smisp WHERE cmpd_id = ?', (key,))
                rsmi = cursor.fetchone()[0]
                print('%s,%s,%s,%s,%s,%s' % (smi, rsmi, id, value[0], value[1], value[2]))
        elif size_of_change <= max_size:
            cursor.execute('SELECT smiles FROM cmpd_smisp WHERE cmpd_id = ?', (key,))
            rsmi = cursor.fetchone()[0]
            print('%s,%s,%s,%s,%s,%s' % (search_string, rsmi, id, value[0], value[1], value[2]))