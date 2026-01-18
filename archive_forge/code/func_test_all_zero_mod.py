from cirq.linalg.tolerance import all_near_zero, all_near_zero_mod, near_zero, near_zero_mod
def test_all_zero_mod():
    atol = 5
    assert all_near_zero_mod(0, 100, atol=atol)
    assert all_near_zero_mod(4.5, 100, atol=atol)
    assert not all_near_zero_mod(5.5, 100, atol=atol)
    assert all_near_zero_mod(100, 100, atol=atol)
    assert all_near_zero_mod(95.5, 100, atol=atol)
    assert not all_near_zero_mod(94.5, 100, atol=atol)
    assert all_near_zero_mod(-4.5, 100, atol=atol)
    assert not all_near_zero_mod(-5.5, 100, atol=atol)
    assert all_near_zero_mod(104.5, 100, atol=atol)
    assert not all_near_zero_mod(105.5, 100, atol=atol)
    assert all_near_zero_mod([-4.5, 0, 1, 4.5, 3, 95.5, 104.5], 100, atol=atol)
    assert not all_near_zero_mod([-4.5, 0, 1, 4.5, 30], 100, atol=atol)