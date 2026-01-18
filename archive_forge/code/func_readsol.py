from .core import LpSolver_CMD, LpSolver, subprocess, PulpSolverError, clock, log
from .. import constants
import os
import warnings
def readsol(self, filename):
    """Read a CPLEX solution file"""
    try:
        import xml.etree.ElementTree as et
    except ImportError:
        import elementtree.ElementTree as et
    solutionXML = et.parse(filename).getroot()
    solutionheader = solutionXML.find('header')
    statusString = solutionheader.get('solutionStatusString')
    statusValue = solutionheader.get('solutionStatusValue')
    cplexStatus = {'1': constants.LpStatusOptimal, '101': constants.LpStatusOptimal, '102': constants.LpStatusOptimal, '104': constants.LpStatusOptimal, '105': constants.LpStatusOptimal, '107': constants.LpStatusOptimal, '109': constants.LpStatusOptimal, '113': constants.LpStatusOptimal}
    if statusValue not in cplexStatus:
        raise PulpSolverError("Unknown status returned by CPLEX: \ncode: '{}', string: '{}'".format(statusValue, statusString))
    status = cplexStatus[statusValue]
    cplexSolStatus = {'104': constants.LpSolutionIntegerFeasible, '105': constants.LpSolutionIntegerFeasible, '107': constants.LpSolutionIntegerFeasible, '109': constants.LpSolutionIntegerFeasible, '111': constants.LpSolutionIntegerFeasible, '113': constants.LpSolutionIntegerFeasible}
    solStatus = cplexSolStatus.get(statusValue)
    shadowPrices = {}
    slacks = {}
    constraints = solutionXML.find('linearConstraints')
    for constraint in constraints:
        name = constraint.get('name')
        slack = constraint.get('slack')
        shadowPrice = constraint.get('dual')
        try:
            shadowPrices[name] = float(shadowPrice)
        except TypeError:
            shadowPrices[name] = None
        slacks[name] = float(slack)
    values = {}
    reducedCosts = {}
    for variable in solutionXML.find('variables'):
        name = variable.get('name')
        value = variable.get('value')
        values[name] = float(value)
        reducedCost = variable.get('reducedCost')
        try:
            reducedCosts[name] = float(reducedCost)
        except TypeError:
            reducedCosts[name] = None
    return (status, values, reducedCosts, shadowPrices, slacks, solStatus)