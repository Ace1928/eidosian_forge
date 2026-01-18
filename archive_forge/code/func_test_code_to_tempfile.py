import sys
from dill.temp import dump, dump_source, dumpIO, dumpIO_source
from dill.temp import load, load_source, loadIO, loadIO_source
def test_code_to_tempfile():
    if not WINDOWS:
        pyfile = dump_source(f, alias='_f')
        _f = load_source(pyfile)
        assert _f(4) == f(4)