def test_nomad(datadir):
    from ase.io import iread
    path = datadir / 'nomad-images.nomad-json'
    images = list(iread(path))
    assert len(images) == 3
    for atoms in images:
        assert all(atoms.pbc)
        assert (atoms.cell > 0).sum() == 3
        assert atoms.get_chemical_formula() == 'As24Sr32'