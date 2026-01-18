from rdkit.Dbase import DbConnection
from rdkit.ML.Data import Quantize
def runIt(namesAndTypes, dbConnect, nBounds, resCol, typesToDo=['float']):
    results = map(lambda x: x[0], dbConnect.GetColumns(namesAndTypes[resCol][0]))
    nPossibleRes = max(results) + 1
    for cName, cType in namesAndTypes:
        if cType in typesToDo:
            dList = map(lambda x: x[0], dbConnect.GetColumns(cName))
            qDat = Quantize.FindVarMultQuantBounds(dList, nBounds, results, nPossibleRes)
            print(cName, qDat)