import pytest
@pytest.mark.parametrize('n_cols, n_rows', [(3, None), (None, 3), (3, 3)])
def test_3x3_rl_tb(self, kivy_clock, n_cols, n_rows):
    assert {4: (100, 100), 5: (0, 100), 7: (100, 0)} == self.compute_layout(n_data=8, orientation='rl-tb', n_cols=n_cols, n_rows=n_rows, scroll_to=(50, 50), clock=kivy_clock)