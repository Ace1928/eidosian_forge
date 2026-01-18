from ._constants import *
def addgroup(index, pos):
    if index > state.groups:
        raise s.error('invalid group reference %d' % index, pos)
    if literal:
        literals.append(''.join(literal))
        del literal[:]
    groups.append((len(literals), index))
    literals.append(None)