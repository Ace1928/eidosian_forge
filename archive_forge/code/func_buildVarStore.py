from fontTools import ttLib
from fontTools.ttLib.tables import otTables as ot
def buildVarStore(varRegionList, varDataList):
    self = ot.VarStore()
    self.Format = 1
    self.VarRegionList = varRegionList
    self.VarData = list(varDataList)
    self.VarDataCount = len(self.VarData)
    return self