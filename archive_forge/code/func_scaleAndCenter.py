import copy
import functools
import math
import numpy
from rdkit import Chem
def scaleAndCenter(self, mol, conf, coordCenter=False, canvasSize=None, ignoreHs=False):
    canvasSize = canvasSize or self.canvasSize
    xAccum = 0
    yAccum = 0
    minX = 100000000.0
    minY = 100000000.0
    maxX = -100000000.0
    maxY = -100000000.0
    nAts = mol.GetNumAtoms()
    for i in range(nAts):
        if ignoreHs and mol.GetAtomWithIdx(i).GetAtomicNum() == 1:
            continue
        pos = conf.GetAtomPosition(i) * self.drawingOptions.coordScale
        xAccum += pos[0]
        yAccum += pos[1]
        minX = min(minX, pos[0])
        minY = min(minY, pos[1])
        maxX = max(maxX, pos[0])
        maxY = max(maxY, pos[1])
    dx = abs(maxX - minX)
    dy = abs(maxY - minY)
    xSize = dx * self.currDotsPerAngstrom
    ySize = dy * self.currDotsPerAngstrom
    if coordCenter:
        molTrans = (-xAccum / nAts, -yAccum / nAts)
    else:
        molTrans = (-(minX + (maxX - minX) / 2), -(minY + (maxY - minY) / 2))
    self.molTrans = molTrans
    if xSize >= 0.95 * canvasSize[0]:
        scale = 0.9 * canvasSize[0] / xSize
        xSize *= scale
        ySize *= scale
        self.currDotsPerAngstrom *= scale
        self.currAtomLabelFontSize = max(self.currAtomLabelFontSize * scale, self.drawingOptions.atomLabelMinFontSize)
    if ySize >= 0.95 * canvasSize[1]:
        scale = 0.9 * canvasSize[1] / ySize
        xSize *= scale
        ySize *= scale
        self.currDotsPerAngstrom *= scale
        self.currAtomLabelFontSize = max(self.currAtomLabelFontSize * scale, self.drawingOptions.atomLabelMinFontSize)
    drawingTrans = (canvasSize[0] / 2, canvasSize[1] / 2)
    self.drawingTrans = drawingTrans