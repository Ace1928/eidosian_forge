def test_distmom():
    from ase.dft import get_distribution_moment
    import numpy as np
    precision = 1e-08
    x = np.linspace(-50.0, 50.0, 1000)
    y = np.exp(-x ** 2 / 2.0)
    area, center, mom2 = get_distribution_moment(x, y, (0, 1, 2))
    assert sum((abs(area - np.sqrt(2.0 * np.pi)), abs(center), abs(mom2 - 1.0))) < precision
    x = np.linspace(-1.0, 1.0, 100000)
    for order in range(0, 9):
        y = x ** order
        area = get_distribution_moment(x, y)
        assert abs(area - (1.0 - (-1.0) ** (order + 1)) / (order + 1.0)) < precision
    x = np.linspace(-50.0, 50.0, 100)
    y = np.exp(-2.0 * (x - 7.0) ** 2 / 10.0) + np.exp(-2.0 * (x + 5.0) ** 2 / 10.0)
    center = get_distribution_moment(x, y, 1)
    assert abs(center - 1.0) < precision