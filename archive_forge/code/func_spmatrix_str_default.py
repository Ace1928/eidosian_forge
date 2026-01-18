def spmatrix_str_default(X):
    from sys import maxsize
    from cvxopt.printing import options
    width, height = (options['width'], options['height'])
    iformat, dformat = (options['iformat'], options['dformat'])
    sgn = ['-', '+']
    if X.typecode == 'i':
        fmt = iformat
    else:
        fmt = dformat
    s = ''
    m, n = X.size
    if width < 0:
        width = maxsize
    if height < 0:
        height = maxsize
    if width * height == 0:
        return ''
    rlist = range(0, min(m, height))
    clist = range(0, min(n, width))
    Xr = X[:min(m, height), :min(n, width)]
    Idx = list(zip(*(Xr.I, Xr.J)))
    if len(Idx) > 0:
        if X.typecode == 'z':
            twidth = max([len(fmt % X[i, j].real + sgn[X[i, j].imag > 0] + 'j' + (fmt % abs(X[i, j].imag)).lstrip()) for i in rlist for j in clist])
        else:
            twidth = max([len(fmt % X[i, j]) for i in rlist for j in clist])
    else:
        twidth = 1
    for i in rlist:
        s += '['
        for j in clist:
            if (i, j) in Idx:
                if X.typecode == 'z':
                    s += format(fmt % X[i, j].real + sgn[X[i, j].imag > 0] + 'j' + (fmt % abs(X[i, j].imag)).lstrip(), '>%i' % twidth)
                else:
                    s += format(fmt % X[i, j], '>%i' % twidth)
            else:
                s += format(0, '^%i' % twidth)
            s += ' '
        if width < n:
            s += '... ]\n'
        else:
            s = s[:-1] + ']\n'
    if height < m:
        s += '[' + min(n, width) * (format(':', '^%i' % twidth) + ' ')
        if width < n:
            s += '   ]\n'
        else:
            s = s[:-1] + ']\n'
    return s