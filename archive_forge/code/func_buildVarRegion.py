from fontTools import ttLib
from fontTools.ttLib.tables import otTables as ot
def buildVarRegion(support, axisTags):
    assert all((tag in axisTags for tag in support.keys())), ('Unknown axis tag found.', support, axisTags)
    self = ot.VarRegion()
    self.VarRegionAxis = []
    for tag in axisTags:
        self.VarRegionAxis.append(buildVarRegionAxis(support.get(tag, (0, 0, 0))))
    return self