from ase.lattice import bravais_classes
def polar_space_group(sg):
    sg = validate_space_group(sg)
    pg = get_point_group(sg)
    return pg in ['1', '2', 'm', 'mm2', '4', '4mm', '3', '3m', '6', '6mm']