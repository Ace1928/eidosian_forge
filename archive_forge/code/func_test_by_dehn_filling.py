import re
import string
def test_by_dehn_filling():
    import random
    from snappy import OrientableCuspedCensus
    count = 0
    for M in OrientableCuspedCensus(cusps=3):
        for i in range(20):
            unfilled = random.randint(0, 2)
            for c in range(3):
                if c != unfilled:
                    fillings = [(1, 0), (0, 1), (11, 12), (-13, 16), (9, -11), (8, 9), (1, 7), (13, 14), (14, -15), (17, -18)]
                    M.dehn_fill(fillings[random.randint(0, len(fillings) - 1)], c)
            if 'positive' in M.solution_type():
                count += 1
                helper_test_by_dehn_filling(M)
    print('Tested %d randomly Dehn filled manifolds' % count)