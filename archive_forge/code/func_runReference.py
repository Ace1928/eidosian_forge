import collections
import os
import sys
from time import perf_counter
import numpy as np
import pyqtgraph as pg
from pyqtgraph import configfile
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.parametertree import types as pTypes
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
def runReference(self, nPts):
    clocks = self.clocks
    ref = self.ref
    dt = self.dt
    dur = self.duration
    clocks = clocks.copy()
    for k, v in clocks.items():
        if v is ref:
            del clocks[k]
            break
    ref.refx = 0
    ref.refv = 0
    ref.refm = ref.m0
    ptVals = np.linspace(ref.pt, ref.pt + dt * (nPts - 1), nPts)
    for i in range(1, nPts):
        nextPt = ptVals[i]
        while True:
            tau1, tau2 = ref.accelLimits()
            dtau = min(nextPt - ref.pt, tau2 - ref.pt)
            g = ref.acceleration()
            v, x, t = Simulation.tauStep(dtau, ref.v, ref.x, ref.t, g)
            ref.pt += dtau
            ref.v = v
            ref.x = x
            ref.t = t
            ref.reft = ref.pt
            if ref.pt >= nextPt:
                break
        ref.recordFrame(i)
        for cl in clocks.values():
            while True:
                g = cl.acceleration()
                tau1, tau2 = cl.accelLimits()
                t1 = Simulation.hypIntersect(ref.x, ref.t, ref.v, cl.x, cl.t, cl.v, g)
                dt1 = t1 - cl.t
                v, x, tau = Simulation.hypTStep(dt1, cl.v, cl.x, cl.pt, g)
                if tau < tau1:
                    dtau = tau1 - cl.pt
                    cl.v, cl.x, cl.t = Simulation.tauStep(dtau, cl.v, cl.x, cl.t, g)
                    cl.pt = tau1 - 1e-06
                    continue
                if tau > tau2:
                    dtau = tau2 - cl.pt
                    cl.v, cl.x, cl.t = Simulation.tauStep(dtau, cl.v, cl.x, cl.t, g)
                    cl.pt = tau2
                    continue
                cl.v = v
                cl.x = x
                cl.pt = tau
                cl.t = t1
                cl.m = None
                break
            x = cl.x - ref.x
            t = cl.t - ref.t
            gamma = (1.0 - ref.v ** 2) ** (-0.5)
            vg = -ref.v * gamma
            cl.refx = gamma * (x - ref.v * t)
            cl.reft = ref.pt
            cl.refv = (cl.v - ref.v) / (1.0 - cl.v * ref.v)
            cl.refm = None
            cl.recordFrame(i)
        t += dt