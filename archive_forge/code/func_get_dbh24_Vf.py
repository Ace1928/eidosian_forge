from ase.atoms import Atoms
def get_dbh24_Vf(name):
    """ Returns forward DBH24 TST barrier in kcal/mol
    """
    assert name in dbh24
    d = data[name]
    Vf = d['Vf']
    return Vf