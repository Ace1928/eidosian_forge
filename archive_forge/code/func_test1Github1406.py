import os
import subprocess
import unittest
from rdkit import RDConfig
def test1Github1406(self):
    with open('data/simple.smi') as inf:
        p = subprocess.run(('python', 'rfrag.py'), stdin=inf, stdout=subprocess.PIPE)
    self.assertFalse(p.returncode)
    self.assertEqual(p.stdout, b'c1ccccc1,benzene,,\nCc1ccccc1,toluene,,C[*:1].c1ccc(cc1)[*:1]\n')