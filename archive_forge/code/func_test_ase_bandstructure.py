from ase.lattice import RHL
from pathlib import Path
def test_ase_bandstructure(cli, plt, testdir):
    lat = RHL(3.0, 70.0)
    path = lat.bandpath()
    bs = path.free_electron_band_structure()
    bs_path = Path('bs.json')
    bs.write(bs_path)
    fig_path = Path('bs.png')
    cli.ase('band-structure', str(bs_path), '--output', str(fig_path))
    assert fig_path.is_file()