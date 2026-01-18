from ase.db.table import Table
from types import SimpleNamespace
def test_hide_empty_columns():
    db = TestConnection()
    table = Table(db)
    for show in [True, False]:
        table.select('...', ['a', 'b', 'c'], '', 10, 0, show_empty_columns=show)
        if show:
            assert table.columns == ['a', 'b', 'c']
        else:
            assert table.columns == ['a', 'b']