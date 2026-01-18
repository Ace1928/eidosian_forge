import numpy as np
def array_to_string(array, indent=14):
    """Converts given numpy array to a string, which when printed will pass
    flake8 tests."""
    text = np.array2string(array, separator=', ', suppress_small=False, formatter={'float': '{:.8f}'.format, 'bool': '{}'.format})
    text = ' ' * indent + text.replace('\n', '\n' + ' ' * indent)
    return text