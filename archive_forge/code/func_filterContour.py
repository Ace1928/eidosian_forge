from fontTools.pens.filterPen import ContourFilterPen
def filterContour(self, contour):
    if not contour or contour[0][0] != 'moveTo' or contour[-1][0] != 'closePath' or (len(contour) < 3):
        return
    movePt = contour[0][1][0]
    lastSeg = contour[-2][1]
    if lastSeg and movePt != lastSeg[-1]:
        contour[-1:] = [('lineTo', (movePt,)), ('closePath', ())]