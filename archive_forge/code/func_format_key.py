from ase.calculators.calculator import Parameters
def format_key(key):
    """ Fix the fdf-key replacing '_' with '.' and '__' with '_' """
    key = key.replace('__', '#')
    key = key.replace('_', '.')
    key = key.replace('#', '_')
    return key