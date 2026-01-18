def test_jsondb():
    """Read and write json from/to file descriptor."""
    from io import StringIO
    from ase.io import read, write
    s = u'\n    {"1":\n         {"numbers": [1, 1],\n          "positions": [[0.0, 0.0, 0.35],\n                        [0.0, 0.0, -0.35]]}}\n    '
    fd = StringIO(s)
    a = read(fd, format='json')
    assert a.get_chemical_formula() == 'H2'
    fd = StringIO()
    write(fd, a, format='json')
    fd.seek(0)
    a = read(fd, format='json')
    assert a.get_chemical_formula() == 'H2'