import ast
import string
def binary2num(bin):
    bin = strjoin(bin.split())
    l = 0
    for digit in bin:
        l = l << 1
        if digit != '0':
            l = l | 1
    return l