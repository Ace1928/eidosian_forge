from __future__ import print_function
import argparse
import sys
import graphviz
from ._discover import findMachines
def tableMaker(inputLabel, outputLabels, port, _E=elementMaker):
    """
    Construct an HTML table to label a state transition.
    """
    colspan = {}
    if outputLabels:
        colspan['colspan'] = str(len(outputLabels))
    inputLabelCell = _E('td', _E('font', inputLabel, face='menlo-italic'), color='purple', port=port, **colspan)
    pointSize = {'point-size': '9'}
    outputLabelCells = [_E('td', _E('font', outputLabel, **pointSize), color='pink') for outputLabel in outputLabels]
    rows = [_E('tr', inputLabelCell)]
    if outputLabels:
        rows.append(_E('tr', *outputLabelCells))
    return _E('table', *rows)