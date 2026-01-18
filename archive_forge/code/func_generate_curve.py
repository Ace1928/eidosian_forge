from .cu2qu import *
import random
import timeit
def generate_curve():
    return [tuple((float(random.randint(0, 2048)) for coord in range(2))) for point in range(4)]