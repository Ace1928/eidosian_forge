from subprocess import Popen, PIPE
from distutils import spawn
import os
import math
import random
import time
import sys
import platform
def getAvailability(GPUs, maxLoad=0.5, maxMemory=0.5, memoryFree=0, includeNan=False, excludeID=[], excludeUUID=[]):
    GPUavailability = [1 if gpu.memoryFree >= memoryFree and (gpu.load < maxLoad or (includeNan and math.isnan(gpu.load))) and (gpu.memoryUtil < maxMemory or (includeNan and math.isnan(gpu.memoryUtil))) and (gpu.id not in excludeID and gpu.uuid not in excludeUUID) else 0 for gpu in GPUs]
    return GPUavailability