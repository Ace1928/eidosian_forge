from fontTools.cffLib import maxStackLimit
def specializeProgram(program, getNumRegions=None, **kwargs):
    return commandsToProgram(specializeCommands(programToCommands(program, getNumRegions), **kwargs))