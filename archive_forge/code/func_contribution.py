import numpy
def contribution(i):
    return hv.hypervolume(numpy.concatenate((wobj[:i], wobj[i + 1:])), ref)