import numpy as np
from ase.neighborlist import NeighborList
from ase.data import atomic_masses, chemical_symbols
from ase import Atoms
def load_xyz_input_gpumd(fd, species=None, isotope_masses=None):
    """
    Read the structure input file for GPUMD and return an ase Atoms object
    togehter with a dictionary with parameters and a types-to-symbols map

    Parameters
    ----------
    fd : file | str
        File object or name of file from which to read the Atoms object
    species : List[str]
        List with the chemical symbols that correspond to each type, will take
        precedence over isotope_masses
    isotope_masses: Dict[str, List[float]]
        Dictionary with chemical symbols and lists of the associated atomic
        masses, which is used to identify the chemical symbols that correspond
        to the types not found in species_types. The default is to find the
        closest match :data:`ase.data.atomic_masses`.

    Returns
    -------
    atoms : Atoms
        Atoms object
    input_parameters : Dict[str, int]
        Dictionary with parameters from the first row of the input file, namely
        'N', 'M', 'cutoff', 'triclinic', 'has_velocity' and 'num_of_groups'
    species : List[str]
        List with the chemical symbols that correspond to each type

    Raises
    ------
    ValueError
        Raised if the list of species is incompatible with the input file
    """
    first_line = next(fd)
    print(first_line)
    input_parameters = {}
    keys = ['N', 'M', 'cutoff', 'triclinic', 'has_velocity', 'num_of_groups']
    types = [float if key == 'cutoff' else int for key in keys]
    for k, (key, typ) in enumerate(zip(keys, types)):
        input_parameters[key] = typ(first_line.split()[k])
    second_line = next(fd)
    second_arr = np.array(second_line.split())
    pbc = second_arr[:3].astype(bool)
    if input_parameters['triclinic']:
        cell = second_arr[3:].astype(float).reshape((3, 3))
    else:
        cell = np.diag(second_arr[3:].astype(float))
    n_rows = input_parameters['N']
    n_columns = 5 + input_parameters['has_velocity'] * 3 + input_parameters['num_of_groups']
    rest_lines = [next(fd) for _ in range(n_rows)]
    rest_arr = np.array([line.split() for line in rest_lines])
    assert rest_arr.shape == (n_rows, n_columns)
    atom_types = rest_arr[:, 0].astype(int)
    positions = rest_arr[:, 1:4].astype(float)
    masses = rest_arr[:, 4].astype(float)
    if species is None:
        type_symbol_map = {}
    if isotope_masses is not None:
        mass_symbols = {mass: symbol for symbol, masses in isotope_masses.items() for mass in masses}
    symbols = []
    for atom_type, mass in zip(atom_types, masses):
        if species is None:
            if atom_type not in type_symbol_map:
                if isotope_masses is not None:
                    nearest_value = find_nearest_value(list(mass_symbols.keys()), mass)
                    symbol = mass_symbols[nearest_value]
                else:
                    symbol = chemical_symbols[find_nearest_index(atomic_masses, mass)]
                type_symbol_map[atom_type] = symbol
            else:
                symbol = type_symbol_map[atom_type]
        else:
            if atom_type > len(species):
                raise Exception('There is no entry for atom type {} in the species list!'.format(atom_type))
            symbol = species[atom_type]
        symbols.append(symbol)
    if species is None:
        species = [type_symbol_map[i] for i in sorted(type_symbol_map.keys())]
    atoms = Atoms(symbols=symbols, positions=positions, masses=masses, pbc=pbc, cell=cell)
    if input_parameters['has_velocity']:
        velocities = rest_arr[:, 5:8].astype(float)
        atoms.set_velocities(velocities)
    if input_parameters['num_of_groups']:
        start_col = 5 + 3 * input_parameters['has_velocity']
        groups = rest_arr[:, start_col:].astype(int)
        atoms.info = {i: {'groups': groups[i, :]} for i in range(n_rows)}
    return (atoms, input_parameters, species)