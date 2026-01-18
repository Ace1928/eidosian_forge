import docutils.utils.math.tex2unichar as tex2unichar
def handle_keyword(name, node, string):
    skip = 0
    if len(string) > 0 and string[0] == ' ':
        string = string[1:]
        skip = 1
    if name == 'begin':
        if not string.startswith('{matrix}'):
            raise SyntaxError('Environment not supported! Supported environment: "matrix".')
        skip += 8
        entry = mtd()
        table = mtable(mtr(entry))
        node.append(table)
        node = entry
    elif name == 'end':
        if not string.startswith('{matrix}'):
            raise SyntaxError('Expected "\\end{matrix}"!')
        skip += 8
        node = node.close().close().close()
    elif name in ('text', 'mathrm'):
        if string[0] != '{':
            raise SyntaxError('Expected "\\text{...}"!')
        i = string.find('}')
        if i == -1:
            raise SyntaxError('Expected "\\text{...}"!')
        node = node.append(mtext(string[1:i]))
        skip += i + 1
    elif name == 'sqrt':
        sqrt = msqrt()
        node.append(sqrt)
        node = sqrt
    elif name == 'frac':
        frac = mfrac()
        node.append(frac)
        node = frac
    elif name == 'left':
        for par in ['(', '[', '|', '\\{', '\\langle', '.']:
            if string.startswith(par):
                break
        else:
            raise SyntaxError('Missing left-brace!')
        fenced = mfenced(par)
        node.append(fenced)
        row = mrow()
        fenced.append(row)
        node = row
        skip += len(par)
    elif name == 'right':
        for par in [')', ']', '|', '\\}', '\\rangle', '.']:
            if string.startswith(par):
                break
        else:
            raise SyntaxError('Missing right-brace!')
        node = node.close()
        node.closepar = par
        node = node.close()
        skip += len(par)
    elif name == 'not':
        for operator in negatables:
            if string.startswith(operator):
                break
        else:
            raise SyntaxError('Expected something to negate: "\\not ..."!')
        node = node.append(mo(negatables[operator]))
        skip += len(operator)
    elif name == 'mathbf':
        style = mstyle(nchildren=1, fontweight='bold')
        node.append(style)
        node = style
    elif name == 'mathbb':
        if string[0] != '{' or not string[1].isupper() or string[2] != '}':
            raise SyntaxError('Expected something like "\\mathbb{A}"!')
        node = node.append(mi(mathbb[string[1]]))
        skip += 3
    elif name in ('mathscr', 'mathcal'):
        if string[0] != '{' or string[2] != '}':
            raise SyntaxError('Expected something like "\\mathscr{A}"!')
        node = node.append(mi(mathscr[string[1]]))
        skip += 3
    elif name == 'colon':
        node = node.append(mo(':'))
    elif name in Greek:
        node = node.append(mo(Greek[name]))
    elif name in letters:
        node = node.append(mi(letters[name]))
    elif name in special:
        node = node.append(mo(special[name]))
    elif name in functions:
        node = node.append(mo(name))
    elif name in over:
        ovr = mover(mo(over[name]), reversed=True)
        node.append(ovr)
        node = ovr
    else:
        raise SyntaxError('Unknown LaTeX command: ' + name)
    return (node, skip)