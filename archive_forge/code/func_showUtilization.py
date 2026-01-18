from subprocess import Popen, PIPE
from distutils import spawn
import os
import math
import random
import time
import sys
import platform
def showUtilization(all=False, attrList=None, useOldCode=False):
    GPUs = getGPUs()
    if all:
        if useOldCode:
            print(' ID | Name | Serial | UUID || GPU util. | Memory util. || Memory total | Memory used | Memory free || Display mode | Display active |')
            print('------------------------------------------------------------------------------------------------------------------------------')
            for gpu in GPUs:
                print(' {0:2d} | {1:s}  | {2:s} | {3:s} || {4:3.0f}% | {5:3.0f}% || {6:.0f}MB | {7:.0f}MB | {8:.0f}MB || {9:s} | {10:s}'.format(gpu.id, gpu.name, gpu.serial, gpu.uuid, gpu.load * 100, gpu.memoryUtil * 100, gpu.memoryTotal, gpu.memoryUsed, gpu.memoryFree, gpu.display_mode, gpu.display_active))
        else:
            attrList = [[{'attr': 'id', 'name': 'ID'}, {'attr': 'name', 'name': 'Name'}, {'attr': 'serial', 'name': 'Serial'}, {'attr': 'uuid', 'name': 'UUID'}], [{'attr': 'temperature', 'name': 'GPU temp.', 'suffix': 'C', 'transform': lambda x: x, 'precision': 0}, {'attr': 'load', 'name': 'GPU util.', 'suffix': '%', 'transform': lambda x: x * 100, 'precision': 0}, {'attr': 'memoryUtil', 'name': 'Memory util.', 'suffix': '%', 'transform': lambda x: x * 100, 'precision': 0}], [{'attr': 'memoryTotal', 'name': 'Memory total', 'suffix': 'MB', 'precision': 0}, {'attr': 'memoryUsed', 'name': 'Memory used', 'suffix': 'MB', 'precision': 0}, {'attr': 'memoryFree', 'name': 'Memory free', 'suffix': 'MB', 'precision': 0}], [{'attr': 'display_mode', 'name': 'Display mode'}, {'attr': 'display_active', 'name': 'Display active'}]]
    elif useOldCode:
        print(' ID  GPU  MEM')
        print('--------------')
        for gpu in GPUs:
            print(' {0:2d} {1:3.0f}% {2:3.0f}%'.format(gpu.id, gpu.load * 100, gpu.memoryUtil * 100))
    else:
        attrList = [[{'attr': 'id', 'name': 'ID'}, {'attr': 'load', 'name': 'GPU', 'suffix': '%', 'transform': lambda x: x * 100, 'precision': 0}, {'attr': 'memoryUtil', 'name': 'MEM', 'suffix': '%', 'transform': lambda x: x * 100, 'precision': 0}]]
    if not useOldCode:
        if attrList is not None:
            headerString = ''
            GPUstrings = [''] * len(GPUs)
            for attrGroup in attrList:
                for attrDict in attrGroup:
                    headerString = headerString + '| ' + attrDict['name'] + ' '
                    headerWidth = len(attrDict['name'])
                    minWidth = len(attrDict['name'])
                    attrPrecision = '.' + str(attrDict['precision']) if 'precision' in attrDict.keys() else ''
                    attrSuffix = str(attrDict['suffix']) if 'suffix' in attrDict.keys() else ''
                    attrTransform = attrDict['transform'] if 'transform' in attrDict.keys() else lambda x: x
                    for gpu in GPUs:
                        attr = getattr(gpu, attrDict['attr'])
                        attr = attrTransform(attr)
                        if isinstance(attr, float):
                            attrStr = ('{0:' + attrPrecision + 'f}').format(attr)
                        elif isinstance(attr, int):
                            attrStr = '{0:d}'.format(attr)
                        elif isinstance(attr, str):
                            attrStr = attr
                        elif sys.version_info[0] == 2:
                            if isinstance(attr, unicode):
                                attrStr = attr.encode('ascii', 'ignore')
                        else:
                            raise TypeError('Unhandled object type (' + str(type(attr)) + ") for attribute '" + attrDict['name'] + "'")
                        attrStr += attrSuffix
                        minWidth = max(minWidth, len(attrStr))
                    headerString += ' ' * max(0, minWidth - headerWidth)
                    minWidthStr = str(minWidth - len(attrSuffix))
                    for gpuIdx, gpu in enumerate(GPUs):
                        attr = getattr(gpu, attrDict['attr'])
                        attr = attrTransform(attr)
                        if isinstance(attr, float):
                            attrStr = ('{0:' + minWidthStr + attrPrecision + 'f}').format(attr)
                        elif isinstance(attr, int):
                            attrStr = ('{0:' + minWidthStr + 'd}').format(attr)
                        elif isinstance(attr, str):
                            attrStr = ('{0:' + minWidthStr + 's}').format(attr)
                        elif sys.version_info[0] == 2:
                            if isinstance(attr, unicode):
                                attrStr = ('{0:' + minWidthStr + 's}').format(attr.encode('ascii', 'ignore'))
                        else:
                            raise TypeError('Unhandled object type (' + str(type(attr)) + ") for attribute '" + attrDict['name'] + "'")
                        attrStr += attrSuffix
                        GPUstrings[gpuIdx] += '| ' + attrStr + ' '
                headerString = headerString + '|'
                for gpuIdx, gpu in enumerate(GPUs):
                    GPUstrings[gpuIdx] += '|'
            headerSpacingString = '-' * len(headerString)
            print(headerString)
            print(headerSpacingString)
            for GPUstring in GPUstrings:
                print(GPUstring)