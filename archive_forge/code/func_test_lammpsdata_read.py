import ase.io
from .parse_lammps_data_file import lammpsdata_file_extracted_sections
from .comparison import compare_with_pytest_approx
def test_lammpsdata_read(lammpsdata_file_path):
    atoms = ase.io.read(lammpsdata_file_path, format='lammps-data', units='metal')
    expected_values = lammpsdata_file_extracted_sections(lammpsdata_file_path)
    cell_read_in = atoms.get_cell()
    cell_expected = expected_values['cell']
    compare_with_pytest_approx(cell_read_in, cell_expected, REL_TOL)
    masses_read_in = atoms.get_masses()
    masses_expected = [expected_values['mass']] * len(expected_values['positions'])
    compare_with_pytest_approx(masses_read_in, masses_expected, REL_TOL)
    positions_read_in = atoms.get_positions()
    positions_expected = expected_values['positions']
    compare_with_pytest_approx(positions_read_in, positions_expected, REL_TOL)
    velocities_read_in = atoms.get_velocities()
    velocities_expected = expected_values['velocities']
    compare_with_pytest_approx(velocities_read_in, velocities_expected, REL_TOL)