from ase.atoms import Atoms
def get_dbh24_tst(name):
    """ Returns DBH24 TST names
    """
    assert name in dbh24_reaction_list
    d = dbh24_reaction_list[name]
    tst = d['tst']
    return tst