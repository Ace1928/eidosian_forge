import snappy
import snappy.snap.t3mlite as t3m
def cusped_test():
    for M in snappy.OrientableCuspedCensus[:10]:
        T = t3m.Mcomplex(M)
        T.find_normal_surfaces()