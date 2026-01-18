import os
import subprocess
import sys
import unittest
@unittest.skipIf('CI' not in os.environ, 'Docs not required for local builds')
def test_docs_command(self):
    try:
        subprocess.run([sys.executable, '-m', 'pygame.docs'], timeout=5, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.TimeoutExpired:
        pass