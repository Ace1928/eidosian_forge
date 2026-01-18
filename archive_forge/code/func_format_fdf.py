from ase.calculators.calculator import Parameters
def format_fdf(key, value):
    """
    Write an fdf key-word value pair.

    Parameters:
        - key   : The fdf-key
        - value : The fdf value.
    """
    if isinstance(value, (list, tuple)) and len(value) == 0:
        return ''
    key = format_key(key)
    new_value = format_value(value)
    if isinstance(value, list):
        string = '%block ' + key + '\n' + new_value + '\n' + '%endblock ' + key + '\n'
    else:
        string = '%s\t%s\n' % (key, new_value)
    return string