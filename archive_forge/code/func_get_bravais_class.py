from ase.lattice import bravais_classes
def get_bravais_class(sg):
    sg = validate_space_group(sg)
    pearson_symbol = _crystal_family[sg] + _lattice_centering[sg]
    return bravais_classes[pearson_symbol]