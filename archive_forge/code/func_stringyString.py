def stringyString(object, indentation=''):
    """
    Expansive string formatting for sequence types.

    C{list.__str__} and C{dict.__str__} use C{repr()} to display their
    elements.  This function also turns these sequence types
    into strings, but uses C{str()} on their elements instead.

    Sequence elements are also displayed on separate lines, and nested
    sequences have nested indentation.
    """
    braces = ''
    sl = []
    if type(object) is dict:
        braces = '{}'
        for key, value in object.items():
            value = stringyString(value, indentation + '   ')
            if isMultiline(value):
                if endsInNewline(value):
                    value = value[:-len('\n')]
                sl.append(f'{indentation} {key}:\n{value}')
            else:
                sl.append(f'{indentation} {key}: {value[len(indentation) + 3:]}')
    elif type(object) is tuple or type(object) is list:
        if type(object) is tuple:
            braces = '()'
        else:
            braces = '[]'
        for element in object:
            element = stringyString(element, indentation + ' ')
            sl.append(element.rstrip() + ',')
    else:
        sl[:] = map(lambda s, i=indentation: i + s, str(object).split('\n'))
    if not sl:
        sl.append(indentation)
    if braces:
        sl[0] = indentation + braces[0] + sl[0][len(indentation) + 1:]
        sl[-1] = sl[-1] + braces[-1]
    s = '\n'.join(sl)
    if isMultiline(s) and (not endsInNewline(s)):
        s = s + '\n'
    return s