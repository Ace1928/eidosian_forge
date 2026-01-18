from __future__ import absolute_import, division, print_function
def getRuleString(deviceType, variableId):
    retVal = variableId + ':'
    if deviceType == 'g8272_cnos':
        if variableId in g8272_cnos:
            retVal = retVal + g8272_cnos[variableId]
        else:
            retVal = 'The variable ' + variableId + ' is not supported'
    elif deviceType == 'g8296_cnos':
        if variableId in g8296_cnos:
            retVal = retVal + g8296_cnos[variableId]
        else:
            retVal = 'The variable ' + variableId + ' is not supported'
    elif deviceType == 'g8332_cnos':
        if variableId in g8332_cnos:
            retVal = retVal + g8332_cnos[variableId]
        else:
            retVal = 'The variable ' + variableId + ' is not supported'
    elif deviceType == 'NE1072T':
        if variableId in NE1072T:
            retVal = retVal + NE1072T[variableId]
        else:
            retVal = 'The variable ' + variableId + ' is not supported'
    elif deviceType == 'NE1032':
        if variableId in NE1032:
            retVal = retVal + NE1032[variableId]
        else:
            retVal = 'The variable ' + variableId + ' is not supported'
    elif deviceType == 'NE1032T':
        if variableId in NE1032T:
            retVal = retVal + NE1032T[variableId]
        else:
            retVal = 'The variable ' + variableId + ' is not supported'
    elif deviceType == 'NE10032':
        if variableId in NE10032:
            retVal = retVal + NE10032[variableId]
        else:
            retVal = 'The variable ' + variableId + ' is not supported'
    elif deviceType == 'NE2572':
        if variableId in NE2572:
            retVal = retVal + NE2572[variableId]
        else:
            retVal = 'The variable ' + variableId + ' is not supported'
    elif deviceType == 'NE0152T':
        if variableId in NE0152T:
            retVal = retVal + NE0152T[variableId]
        else:
            retVal = 'The variable ' + variableId + ' is not supported'
    elif variableId in default_cnos:
        retVal = retVal + default_cnos[variableId]
    else:
        retVal = 'The variable ' + variableId + ' is not supported'
    return retVal