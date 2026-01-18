from io import StringIO
import ase.io
from .parse_lammps_data_file import lammpsdata_file_extracted_sections
from .comparison import compare_with_pytest_approx
def test_lammpsdata_write(atoms):
    lammpsdata_buf = StringIO()
    ase.io.write(lammpsdata_buf, atoms, format='lammps-data', atom_style='full', velocities=True)
    written_values = lammpsdata_file_extracted_sections(lammpsdata_buf)
    cell_written = written_values['cell']
    cell_expected = atoms.get_cell()
    compare_with_pytest_approx(cell_written, cell_expected, REL_TOL)
    positions_written = written_values['positions']
    positions_expected = atoms.get_positions(wrap=True)
    compare_with_pytest_approx(positions_written, positions_expected, REL_TOL)
    velocities_written = written_values['velocities']
    velocities_expected = atoms.get_velocities()
    compare_with_pytest_approx(velocities_written, velocities_expected, REL_TOL)