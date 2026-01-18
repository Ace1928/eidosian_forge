import itertools
import collections
def resource_clock():
    import resource
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime