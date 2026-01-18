from mpmath import *
def test_interval_conversions():
    mp.dps = 15
    iv.dps = 15
    for a, b in ((-0.0, 0), (0.0, 0.5), (1.0, 1), ('-inf', 20.5), ('-inf', float(sqrt(2)))):
        r = mpi(a, b)
        assert int(r.b) == int(b)
        assert float(r.a) == float(a)
        assert float(r.b) == float(b)
        assert complex(r.a) == complex(a)
        assert complex(r.b) == complex(b)