from subprocess import Popen, PIPE
from distutils import spawn
import os
import math
import random
import time
import sys
import platform
def getAvailable(order='first', limit=1, maxLoad=0.5, maxMemory=0.5, memoryFree=0, includeNan=False, excludeID=[], excludeUUID=[]):
    GPUs = getGPUs()
    GPUavailability = getAvailability(GPUs, maxLoad=maxLoad, maxMemory=maxMemory, memoryFree=memoryFree, includeNan=includeNan, excludeID=excludeID, excludeUUID=excludeUUID)
    availAbleGPUindex = [idx for idx in range(0, len(GPUavailability)) if GPUavailability[idx] == 1]
    GPUs = [GPUs[g] for g in availAbleGPUindex]
    if order == 'first':
        GPUs.sort(key=lambda x: float('inf') if math.isnan(x.id) else x.id, reverse=False)
    elif order == 'last':
        GPUs.sort(key=lambda x: float('-inf') if math.isnan(x.id) else x.id, reverse=True)
    elif order == 'random':
        GPUs = [GPUs[g] for g in random.sample(range(0, len(GPUs)), len(GPUs))]
    elif order == 'load':
        GPUs.sort(key=lambda x: float('inf') if math.isnan(x.load) else x.load, reverse=False)
    elif order == 'memory':
        GPUs.sort(key=lambda x: float('inf') if math.isnan(x.memoryUtil) else x.memoryUtil, reverse=False)
    GPUs = GPUs[0:min(limit, len(GPUs))]
    deviceIds = [gpu.id for gpu in GPUs]
    return deviceIds