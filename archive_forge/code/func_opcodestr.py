def opcodestr(opcode):
    if opcode in opcodemap:
        return opcodemap[opcode]
    else:
        return repr(opcode)