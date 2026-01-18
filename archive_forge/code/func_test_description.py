import os
import pytest
import rpy2.robjects as robjects
import rpy2.robjects.help as rh
def test_description(self):
    base_help = rh.Package('base')
    p = base_help.fetch('print')
    d = p.description()
    assert all((isinstance(x, str) for x in d))
    assert len(d) > 0