import re
import ase.io.abinit as io
from ase.calculators.calculator import FileIOCalculator
from subprocess import check_output
def get_abinit_version(command):
    txt = check_output([command, '--version']).decode('ascii')
    m = re.match('\\s*(\\d\\.\\d\\.\\d)', txt)
    if m is None:
        raise RuntimeError('Cannot recognize abinit version. Start of output: {}'.format(txt[:40]))
    return m.group(1)