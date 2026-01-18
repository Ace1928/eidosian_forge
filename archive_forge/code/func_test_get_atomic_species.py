import numpy as np
from ase import io
from ase import build
from ase.io.espresso import parse_position_line
from pytest import approx
def test_get_atomic_species():
    """Parser for atomic species section"""
    from ase.io.espresso import get_atomic_species, read_fortran_namelist
    with open('pw_input.pwi', 'w') as pw_input_f:
        pw_input_f.write(pw_input_text)
    with open('pw_input.pwi', 'r') as pw_input_f:
        data, card_lines = read_fortran_namelist(pw_input_f)
        species_card = get_atomic_species(card_lines, n_species=data['system']['ntyp'])
    assert len(species_card) == 2
    assert species_card[0] == ('H', approx(1.008), 'H.pbe-rrkjus_psl.0.1.UPF')
    assert species_card[1] == ('Fe', approx(55.845), 'Fe.pbe-spn-rrkjus_psl.0.2.1.UPF')