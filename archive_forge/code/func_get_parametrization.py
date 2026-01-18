def get_parametrization(atoms):
    if 'parametrization' in atoms.info['data']:
        return atoms.info['data']['parametrization']
    else:
        raise ValueError('Trying to get the parametrization before it is set!')