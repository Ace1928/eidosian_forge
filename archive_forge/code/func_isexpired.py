from time import time as gettime
def isexpired(self):
    t = gettime()
    return t >= self.weight