from ase.units import Bohr
def read_turbomole_gradient(fd, index=-1):
    """ Method to read turbomole gradient file """
    lines = [x.strip() for x in fd.readlines()]
    start = end = -1
    for i, line in enumerate(lines):
        if not line.startswith('$'):
            continue
        if line.split()[0] == '$grad':
            start = i
        elif start >= 0:
            end = i
            break
    if end <= start:
        raise RuntimeError("File does not contain a valid '$grad' section")
    del lines[:start + 1]
    del lines[end - 1 - start:]
    from ase import Atoms, Atom
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.units import Bohr, Hartree
    images = []
    while lines:
        fields = lines[0].split('=')
        try:
            energy = float(fields[2].split()[0]) * Hartree
        except (IndexError, ValueError) as e:
            raise TurbomoleFormatError() from e
        atoms = Atoms()
        forces = []
        for line in lines[1:]:
            fields = line.split()
            if len(fields) == 4:
                try:
                    symbol = fields[3].lower().capitalize()
                    if symbol == 'Q':
                        symbol = 'X'
                    position = tuple([Bohr * float(x) for x in fields[0:3]])
                except ValueError as e:
                    raise TurbomoleFormatError() from e
                atoms.append(Atom(symbol, position))
            elif len(fields) == 3:
                grad = []
                for val in fields[:3]:
                    try:
                        grad.append(-float(val.replace('D', 'E')) * Hartree / Bohr)
                    except ValueError as e:
                        raise TurbomoleFormatError() from e
                forces.append(grad)
            else:
                break
        calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
        atoms.calc = calc
        images.append(atoms)
        del lines[:2 * len(atoms) + 1]
    return images[index]