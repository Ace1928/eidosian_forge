import os
import pytest
import rpy2.robjects as robjects
import rpy2.robjects.help as rh
def test_fetch(self):
    base_help = rh.Package('base')
    f = base_help.fetch('print')
    assert '\\title' in f.sections.keys()