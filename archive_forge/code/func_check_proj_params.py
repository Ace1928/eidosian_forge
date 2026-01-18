def check_proj_params(name, crs, other_args):
    expected = other_args | {f'proj={name}', 'no_defs'}
    proj_params = set(crs.proj4_init.lstrip('+').split(' +'))
    assert expected == proj_params