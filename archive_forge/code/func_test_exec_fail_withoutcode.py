import pytest
from ase.build import bulk, molecule
from ase.io import write
def test_exec_fail_withoutcode(cli, fname):
    cli.ase('exec', fname, expect_fail=True)