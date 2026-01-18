from pathlib import Path
import subprocess
import os
import sys
import wasabi
def test_jupyter():
    env = dict(os.environ)
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f'{WASABI_DIR}{os.pathsep}{env['PYTHONPATH']}'
    else:
        env['PYTHONPATH'] = str(WASABI_DIR)
    subprocess.run([sys.executable, '-m', 'nbconvert', str(TEST_DATA / 'wasabi-test-notebook.ipynb'), '--execute', '--stdout', '--to', 'notebook'], env=env, check=True)