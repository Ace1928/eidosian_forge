from fontTools.cffLib import maxStackLimit
def generalizeCommands(commands, ignoreErrors=False):
    result = []
    mapping = _GeneralizerDecombinerCommandsMap
    for op, args in commands:
        if any([isinstance(arg, list) for arg in args]):
            try:
                args = [n for arg in args for n in (_convertBlendOpToArgs(arg) if isinstance(arg, list) else [arg])]
            except ValueError:
                if ignoreErrors:
                    result.append(('', args))
                    result.append(('', [op]))
                else:
                    raise
        func = getattr(mapping, op, None)
        if not func:
            result.append((op, args))
            continue
        try:
            for command in func(args):
                result.append(command)
        except ValueError:
            if ignoreErrors:
                result.append(('', args))
                result.append(('', [op]))
            else:
                raise
    return result