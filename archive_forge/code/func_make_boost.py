import pathlib
import subprocess
import sys
import os
import argparse
def make_boost(outdir):
    code_gen = pathlib.Path(__file__).parent / '_boost/include/code_gen.py'
    subprocess.run([sys.executable, str(code_gen), '-o', outdir], check=True)