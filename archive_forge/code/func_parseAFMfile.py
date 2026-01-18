import os
def parseAFMfile(filename):
    """Returns an array holding the widths of all characters in the font.
    Ultra-crude parser"""
    alllines = open(filename, 'r').readlines()
    metriclines = []
    between = 0
    for line in alllines:
        if 'endcharmetrics' in line.lower():
            between = 0
            break
        if between:
            metriclines.append(line)
        if 'startcharmetrics' in line.lower():
            between = 1
    widths = [0] * 255
    for line in metriclines:
        chunks = line.split(';')
        c, cid = chunks[0].split()
        wx, width = chunks[1].split()
        widths[int(cid)] = int(width)
    for i in range(len(widths)):
        if widths[i] == 0:
            widths[i] == widths[32]
    return widths