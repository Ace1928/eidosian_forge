def makeUniqueGroupName(name, groupNames, counter=0):
    newName = name
    if counter > 0:
        newName = '%s%d' % (newName, counter)
    if newName in groupNames:
        return makeUniqueGroupName(name, groupNames, counter + 1)
    return newName