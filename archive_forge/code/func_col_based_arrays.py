def col_based_arrays(self):
    numEls = len(self)
    elemBase = []
    startsBase = []
    indBase = []
    lenBase = []
    for i, col in enumerate(self.cols):
        startsBase.append(len(elemBase))
        elemBase.extend(list(self.coldict[col].values()))
        indBase.extend(list(self.coldict[col].keys()))
        lenBase.append(len(elemBase) - startsBase[-1])
    startsBase.append(len(elemBase))
    return (numEls, startsBase, lenBase, indBase, elemBase)