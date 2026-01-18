import os
import re
import sqlite3
import subprocess
import sys
from optparse import OptionParser
from indexing import cansmirk, heavy_atom_count
from rfrag import fragment_mol
def cmpd_not_in_db_mmp_query(in_smi, cmpd_id):
    query_contexts = set()
    cmpd_frags = fragment_mol(in_smi, cmpd_id)
    for row in cmpd_frags:
        row = row.rstrip()
        row_fields = re.split(',', row)
        if row_fields[3].count('.') == 1:
            a, b = row_fields[3].split('.')
            query_contexts.add(a)
            query_contexts.add(b)
        else:
            query_contexts.add(row_fields[3])
    q_string = "','".join(query_contexts)
    q_string = "'%s'" % q_string
    query_sql = '\n    select  c.cmpd_id,\n            c.core_smi,\n            con.context_smi,\n            con.context_size\n    from    core_table c, context_table con\n    where   c.context_id in (select context_id from context_table where context_smi in (%s))\n            and c.context_id = con.context_id' % q_string
    cursor.execute(query_sql)
    results = cursor.fetchall()
    cmpd_size = heavy_atom_count(in_smi)
    print_smallest_change_mmp(results, cmpd_id, cmpd_size)